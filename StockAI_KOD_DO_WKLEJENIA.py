"""StockAI 0.1 - edukacyjny prototyp analiz 30 spolek GPW.
Uruchomienie: py -m streamlit run app.py
Dane: Yahoo Finance (przez yfinance), Google News RSS. TYLKO WLASNE TESTY.
Wynik 0-100 to indeks, NIE skalibrowane prawdopodobienstwo.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
import math

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 21  # Sesje gieldowe, ok. miesiaca, nie dni kalendarzowych.
STOCKS = {
    "CMP.WA": "Comp", "ABS.WA": "Asseco Business Solutions",
    "ASB.WA": "ASBIS", "SNT.WA": "Synektik",
    "ACP.WA": "Asseco Poland", "ALE.WA": "Allegro",
    "DNP.WA": "Dino Polska", "CDR.WA": "CD Projekt",
    "PKN.WA": "Orlen", "PKO.WA": "PKO BP",
    "PEO.WA": "Bank Pekao", "PZU.WA": "PZU",
    "KGH.WA": "KGHM", "LPP.WA": "LPP",
    "CCC.WA": "CCC", "PGE.WA": "PGE",
    "OPL.WA": "Orange Polska", "MBK.WA": "mBank",
    "JSW.WA": "JSW", "XTB.WA": "XTB",
    "BDX.WA": "Budimex", "KRU.WA": "Kruk",
    "TEN.WA": "Ten Square Games", "WPL.WA": "Wirtualna Polska",
    "GPW.WA": "GPW", "NEU.WA": "Neuca",
    "CAR.WA": "Inter Cars", "VRC.WA": "Vercom",
    "CBF.WA": "cyber_Folks", "WAW.WA": "Wawel",
}
# Dane makro. Sa to wskazniki przyblizone (proxy), nie opis przyczynowy.
MACROS = {"SPY": "Gielda USA (SPY)", "PLN=X": "USD/PLN",
          "BZ=F": "Ropa Brent", "HG=F": "Miedz", "^VIX": "Indeks VIX"}
FEATURES = ["r1", "r5", "r21", "r63", "vol20", "ma20", "ma60", "ma200",
            "rsi14", "volume_ratio"] + [f"macro_{k}" for k in MACROS]
HEADLINE_POS = ("rekord", "wzrost", "zysk", "kontrakt", "dywidend", "wyższ",
                "podwyż", "umowa", "surge", "beats", "upgrade")
HEADLINE_NEG = ("strat", "spad", "kara", "opóź", "ryzyk", "pozew",
                "obniż", "slump", "misses", "downgrade")


def number(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except (ValueError, TypeError, OverflowError):
        return None


def col_extract(batch: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if batch is None or batch.empty:
        return pd.DataFrame()
    try:
        if isinstance(batch.columns, pd.MultiIndex):
            if ticker in batch.columns.get_level_values(0):
                d = batch[ticker].copy()
            elif ticker in batch.columns.get_level_values(1):
                d = batch.xs(ticker, level=1, axis=1).copy()
            else:
                return pd.DataFrame()
        else:
            d = batch.copy()
        if "Close" not in d.columns:
            return pd.DataFrame()
        d = d[[c for c in ("Close", "Volume") if c in d.columns]].copy()
        if "Volume" not in d.columns:
            d["Volume"] = 0.0
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        d = d[~d.index.duplicated(keep="last")].sort_index()
        d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
        d["Volume"] = pd.to_numeric(d["Volume"], errors="coerce").fillna(0.0)
        return d.loc[d["Close"].gt(0) & d["Close"].notna()]
    except (ValueError, KeyError, TypeError):
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def download_market_data():
    symbols = list(STOCKS) + list(MACROS)
    # Jeden zbiorczy request pozwala ograniczyc liczbe zapytan.
    batch = yf.download(tickers=symbols, period="3y", interval="1d",
                        group_by="ticker", auto_adjust=False,
                        progress=False, threads=True, timeout=15)
    frames = {s: col_extract(batch, s) for s in symbols}
    return frames


def rsi(close, window=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50) / 100


def features_for_symbol(ticker, prices, macros):
    d = prices.copy()
    p = d["Close"].astype(float)
    v = d["Volume"].astype(float)
    d["r1"] = p.pct_change(1)
    d["r5"] = p.pct_change(5)
    d["r21"] = p.pct_change(21)
    d["r63"] = p.pct_change(63)
    d["vol20"] = d["r1"].rolling(20).std()
    for w in (20, 60, 200):
        d[f"ma{w}"] = p / p.rolling(w).mean() - 1
    d["rsi14"] = rsi(p)
    d["volume_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    for m in MACROS:
        ref = macros.get(m, pd.DataFrame())
        if ref.empty:
            d[f"macro_{m}"] = 0.0
        else:
            # SHIFT(1): nie uzywamy ceny USA / surowcow z konca dnia,
            # ktora moglaby byc nieznana w chwili zamkniecia GPW.
            macro_ret = ref["Close"].pct_change(5, fill_method=None).shift(1)
            d[f"macro_{m}"] = macro_ret.reindex(d.index, method="ffill")
    d["target"] = p.shift(-HORIZON) / p - 1.0
    d["up"] = (d["target"] > 0).astype(int)
    d["ticker"] = ticker
    d["date"] = d.index
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan)
    # Pierwszych 200 sesji nie ma poprawnych srednich kroczacych.
    d = d.dropna(subset=["ma200", "r63", "vol20"])
    d[FEATURES] = d[FEATURES].fillna(0.0)
    return d


def build_panel(frames):
    macros = {s: frames.get(s, pd.DataFrame()) for s in MACROS}
    historical, current = [], []
    usable = {}
    for ticker in STOCKS:
        p = frames.get(ticker, pd.DataFrame())
        if p.empty or len(p) < 310:
            continue
        feat = features_for_symbol(ticker, p, macros)
        if len(feat) < 100:
            continue
        trainable = feat[feat["target"].notna()].copy()
        historical.append(trainable)
        current.append(feat.iloc[[-1]].copy())
        usable[ticker] = p
    if not historical:
        return pd.DataFrame(), pd.DataFrame(), usable
    return (pd.concat(historical, ignore_index=True),
            pd.concat(current, ignore_index=True), usable)


def train_models(historical):
    """Historyczny podzial chronologiczny + embargo na etykiety 21 sesji."""
    dates = sorted(historical["date"].dropna().unique())
    if len(dates) < 130:
        raise ValueError("Za krotka historia do testu modelu.")
    validation_start = pd.Timestamp(dates[-65])
    # 35 dni kalendarzowych to konserwatywny bufor dla prognozy 21 sesji.
    train_mask = historical["date"] < validation_start - pd.Timedelta(days=35)
    val_mask = historical["date"] >= validation_start
    train = historical.loc[train_mask]
    test = historical.loc[val_mask]
    if len(train) < 400 or len(test) < 80 or train["up"].nunique() < 2:
        raise ValueError("Za malo obserwacji / klas do nauczenia modelu.")
    X = train[FEATURES].astype(float)
    y = train["target"].astype(float).clip(-0.6, 0.6)
    clf = RandomForestClassifier(n_estimators=110, max_depth=8,
                                 min_samples_leaf=25, n_jobs=-1,
                                 class_weight="balanced_subsample", random_state=42)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    boost = HistGradientBoostingRegressor(max_iter=100, max_depth=4,
                                          l2_regularization=3.0,
                                          learning_rate=0.045, random_state=42)
    clf.fit(X, train["up"])
    ridge.fit(X, y)
    boost.fit(X, y)
    X_test = test[FEATURES].astype(float)
    pred_p = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
    pred_r = (ridge.predict(X_test) + boost.predict(X_test)) / 2
    base_guess = float(y.mean())
    metrics = {
        "test_rows": len(test),
        "test_from": str(validation_start.date()),
        "hit_rate": float(accuracy_score(test["up"], pred_p >= 0.5)),
        "brier": float(brier_score_loss(test["up"], pred_p)),
        "mae": float(mean_absolute_error(test["target"], pred_r)),
        "mae_baseline": float(mean_absolute_error(test["target"],
                                                   np.full(len(test), base_guess))),
        "actual_up_rate": float(test["up"].mean()),
    }
    return clf, ridge, boost, metrics


def macro_sensitivity(prices, frames):
    changes = prices["Close"].pct_change(fill_method=None)
    relations = {}
    for ticker, label in MACROS.items():
        m = frames.get(ticker, pd.DataFrame())
        if m.empty:
            continue
        # przesuniecie chroni przed uzyciem danych z przyszlych godzin.
        other = m["Close"].pct_change(fill_method=None).shift(1)
        overlap = pd.concat([changes, other.reindex(changes.index, method="ffill")],
                            axis=1).dropna().tail(250)
        if len(overlap) >= 75:
            corr = number(overlap.iloc[:, 0].corr(overlap.iloc[:, 1]))
            if corr is not None:
                relations[label] = corr
    if not relations:
        return 0.0, "brak danych", {}
    best = max(relations, key=lambda key: abs(relations[key]))
    return abs(relations[best]), best, relations


@st.cache_data(ttl=86400, show_spinner=False)
def get_company_info(ticker):
    try:
        info = yf.Ticker(ticker).get_info()
        keep = ["trailingPE", "priceToBook", "debtToEquity", "profitMargins",
                "operatingMargins", "revenueGrowth", "returnOnEquity",
                "targetMeanPrice", "numberOfAnalystOpinions", "targetLowPrice",
                "targetHighPrice", "recommendationKey", "recommendationMean",
                "currency", "shortName", "marketCap"]
        return {k: info.get(k) for k in keep}
    except Exception:
        return {}


@st.cache_data(ttl=10800, show_spinner=False)
def get_news(ticker, name):
    # Nie wchodzimy na strony wydawcow i nie kopiujemy ich tekstow.
    # Z RSS pobieramy tylko naglowek, zrodlo, date i link.
    query = f'"{name}" (GPW OR akcje OR giełda) when:14d'
    url = ("https://news.google.com/rss/search?q=" + quote_plus(query) +
           "&hl=pl&gl=PL&ceid=PL:pl")
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "StockAI-demo/0.1"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        entries = []
        for e in root.findall("./channel/item")[:8]:
            title = (e.findtext("title") or "").strip()
            link = (e.findtext("link") or "").strip()
            source = e.find("source")
            pub = e.findtext("pubDate") or ""
            if not title or not link:
                continue
            try:
                date = parsedate_to_datetime(pub)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                pub = date.astimezone(timezone.utc).date().isoformat()
            except (ValueError, TypeError):
                pub = "brak daty"
            entries.append({"title": title, "url": link,
                            "source": source.text if source is not None else "Google News",
                            "date": pub})
        return entries, url
    except (requests.RequestException, ET.ParseError):
        return [], url


def sentiment_adjustment(items):
    # BARDZO PROSTA HEURYSTYKA, nie profesjonalna analiza nastrojow.
    # Nigdy nie stanowi glownego wejscia do prognozy.
    plus = minus = 0
    for item in items:
        text = item["title"].lower()
        plus += int(any(word in text for word in HEADLINE_POS))
        minus += int(any(word in text for word in HEADLINE_NEG))
    return float(np.clip((plus - minus) * 0.5, -2, 2))


def fundamental_adjustment(info):
    if not info:
        return 0.0, 0
    adjustment, available = 0.0, 0
    for k in ("profitMargins", "debtToEquity", "revenueGrowth"):
        if number(info.get(k)) is not None:
            available += 1
    margin = number(info.get("profitMargins"))
    debt = number(info.get("debtToEquity"))
    growth = number(info.get("revenueGrowth"))
    if margin is not None:
        adjustment += 1.5 if margin > .15 else (-1.5 if margin < 0 else 0)
    if debt is not None:
        adjustment += 1 if debt < 50 else (-2 if debt > 150 else 0)
    if growth is not None:
        adjustment += 1.5 if growth > .10 else (-1.5 if growth < 0 else 0)
    return float(np.clip(adjustment, -5, 5)), available


def evaluate_current(current, models, historical, frames, usable, contexts):
    clf, ridge, boost, _ = models
    X = current[FEATURES].astype(float)
    prob = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    r_ridge = ridge.predict(X)
    r_boost = boost.predict(X)
    ret = np.clip(0.5 * (r_ridge + r_boost), -.35, .35)
    dispersion = np.abs(r_ridge - r_boost)
    # 0..100 indeks, NIE empiryczne 0..100% prawdopodobienstwa.
    base_score = 100 * (.65 * prob + .35 * (.5 + .5 * np.tanh(ret / .10)))
    results = []
    for i, row in current.reset_index(drop=True).iterrows():
        ticker = row["ticker"]
        info, news, _ = contexts.get(ticker, ({}, [], ""))
        fund_adj, fundamental_count = fundamental_adjustment(info)
        news_adj = sentiment_adjustment(news)
        score = int(np.clip(round(base_score[i] + fund_adj + news_adj), 0, 100))
        price = float(row["Close"])
        macro_strength, macro_name, relations = macro_sensitivity(usable[ticker], frames)
        results.append({
            "Spółka": STOCKS[ticker], "Ticker": ticker, "Wynik AI": score,
            "Kurs (zł)": round(price, 2), "Prognoza 21 sesji (zł)": round(price * (1 + ret[i]), 2),
            "Zmiana modelu (%)": round(float(ret[i]) * 100, 2),
            "Sygnał ML (%)": round(float(prob[i]) * 100, 1),
            "Rozbieżność modeli (pkt proc.)": round(float(dispersion[i]) * 100, 2),
            "Wrażliwość makro": round(float(macro_strength), 3),
            "Dominujący czynnik": macro_name,
            "Dostępne wskaźniki fundamentów": fundamental_count,
            "Wiadomości": len(news), "Korekta fundamentów": fund_adj,
            "Korekta nagłówków": news_adj,
            "Data notowań": pd.Timestamp(row["date"]).date().isoformat(),
            "Korelacje": relations,
        })
    result = pd.DataFrame(results)
    # 5 spolek o najwyzszej wzglednej korelacji z czynnikami zewnetrznymi.
    macro_ids = result.nlargest(min(5, len(result)), "Wrażliwość makro").index
    result["Grupa"] = "Środek / niepewne"
    result.loc[macro_ids, "Grupa"] = "5 najbardziej wrażliwych na makro"
    remainder = result.drop(index=macro_ids).sort_values("Wynik AI", ascending=False)
    result.loc[remainder.index[:10], "Grupa"] = "10 najwyższych sygnałów"
    result.loc[remainder.index[-10:], "Grupa"] = "10 najniższych sygnałów"
    return result.sort_values("Wynik AI", ascending=False).reset_index(drop=True)


def fetch_contexts(tickers):
    contexts = {}
    def task(t):
        info = get_company_info(t)
        news, url = get_news(t, STOCKS[t])
        return info, news, url
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(task, t): t for t in tickers}
        progress = st.progress(0, text="Pobieram fundamenty i najnowsze nagłówki…")
        for i, future in enumerate(as_completed(futures), start=1):
            t = futures[future]
            try:
                contexts[t] = future.result()
            except Exception:
                contexts[t] = ({}, [], "")
            progress.progress(i / len(tickers), text=f"Sprawdzone firmy: {i}/{len(tickers)}")
        progress.empty()
    return contexts


def show_optional(x, is_percent=False, unit=""):
    v = number(x)
    if v is None:
        return "brak danych"
    return f"{100*v:.1f}%" if is_percent else f"{v:,.2f}{unit}".replace(",", " ")


def main():
    st.set_page_config(page_title="StockAI | GPW", page_icon="📈", layout="wide")
    st.title("📈 StockAI — 30 spółek GPW")
    st.caption("Dane automatyczne • prognoza 21 sesji • model matematyczny + uczenie maszynowe • indeks 0–100")
    st.warning("PROTOTYP EDUKACYJNY. Wynik 0–100 NIE jest gwarancją ani skalibrowanym "
               "prawdopodobieństwem. Prognozowane ceny nie są wyceną fundamentalną ani poradą inwestycyjną.")
    with st.sidebar:
        st.header("Sterowanie")
        if st.button("🔄 Odśwież wszystkie dane"):
            download_market_data.clear()
            get_company_info.clear()
            get_news.clear()
            st.rerun()
        st.write("**Horyzont:** 21 sesji GPW (~miesiąc)")
        st.write("**Liczba planowanych spółek:** 30")
        st.write("Źródła testowe: [Yahoo Finance](https://finance.yahoo.com/), "
                 "[Google News RSS](https://news.google.com/).")
        st.caption("Wersja do użytku własnego. Źródła nie stanowią licencji do sprzedaży danych.")
    try:
        with st.spinner("Pobieram 3 lata notowań GPW i wskaźników zewnętrznych…"):
            frames = download_market_data()
        panel, current, usable = build_panel(frames)
        if current.empty:
            st.error("Brak wystarczających notowań. Yahoo może chwilowo blokować zapytania. "
                     "Sprawdź internet i spróbuj ponownie później. Nie wyświetlam zmyślonych prognoz.")
            st.stop()
        with st.spinner("Trenuję 3 modele i testuję je na historycznym okresie…"):
            models = train_models(panel)
        contexts = fetch_contexts(list(usable))
        predictions = evaluate_current(current, models, panel, frames, usable, contexts)
    except Exception as error:
        st.error(f"Nie udało się zakończyć analizy: {error}")
        st.info("Sprawdź połączenie, aktualność bibliotek lub spróbuj przyciskiem odświeżania.")
        st.stop()

    missing = [f"{STOCKS[x]} ({x})" for x in STOCKS if x not in usable]
    a, b, c, d = st.columns(4)
    a.metric("Spółki z danymi", f"{len(predictions)}/30")
    b.metric("Średni indeks", f"{predictions['Wynik AI'].mean():.0f}/100")
    c.metric("Model kierunku: trafność testu", f"{models[3]['hit_rate']:.1%}")
    d.metric("Błąd prognozy (MAE)", f"{models[3]['mae']:.1%}")
    st.caption(f"Test od {models[3]['test_from']} • {models[3]['test_rows']} obserwacji. "
               "Historyczne wyniki NIE gwarantują przyszłych osiągów.")
    missing_macro = [MACROS[s] for s in MACROS if frames.get(s, pd.DataFrame()).empty]
    if missing_macro:
        st.warning("Brakujące dane makro (model używa dla nich wartości neutralnych): "
                   + ", ".join(missing_macro))
    latest_date = pd.to_datetime(predictions["Data notowań"]).max()
    if (pd.Timestamp.today().normalize() - latest_date).days > 7:
        st.error("UWAGA: ostatnie uzyskane notowania są starsze niż 7 dni. "
                 "Nie traktuj prognoz jako aktualnych.")
    if missing:
        st.warning("Spółki pominięte z powodu braku danych lub zbyt krótkiej historii: "
                   + ", ".join(missing))
    st.subheader("Ranking AI: 0–100")
    st.caption("Wysoki wynik oznacza silniejszy sygnał wzrostowy względem pozostałych danych. "
               "Podział 10/10/5/5 to grupy porównawcze, NIE pewne wzrosty/spadki.")
    options = ["Wszystkie", "10 najwyższych sygnałów", "10 najniższych sygnałów",
               "Środek / niepewne", "5 najbardziej wrażliwych na makro"]
    selected_group = st.selectbox("Pokaż grupę", options)
    display = predictions if selected_group == "Wszystkie" else predictions[predictions["Grupa"] == selected_group]
    visible = ["Spółka", "Ticker", "Wynik AI", "Kurs (zł)", "Prognoza 21 sesji (zł)",
               "Zmiana modelu (%)", "Wrażliwość makro", "Grupa"]
    st.dataframe(display[visible], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Zapisz ranking do CSV", data=predictions.drop(columns=["Korelacje"]).to_csv(index=False).encode("utf-8-sig"),
                       file_name="stockai_ranking.csv", mime="text/csv")
    st.subheader("Szczegóły wybranej spółki")
    tick = st.selectbox("Wybierz spółkę", predictions["Ticker"].tolist(),
                        format_func=lambda t: f"{STOCKS[t]} ({t})")
    row = predictions.set_index("Ticker").loc[tick]
    info, news, news_url = contexts[tick]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wynik AI", f"{row['Wynik AI']}/100")
    c2.metric("Kurs", f"{row['Kurs (zł)']:.2f} zł")
    c3.metric("Model za 21 sesji", f"{row['Prognoza 21 sesji (zł)']:.2f} zł",
              f"{row['Zmiana modelu (%)']:+.2f}%")
    c4.metric("Wskaźnik modelu kierunku", f"{row['Sygnał ML (%)']:.1f}%")
    st.caption(f"Data ostatniego kursu: {row['Data notowań']}. "
               "Sygnał ML to niekalibrowana estymacja klasyfikatora; nie czytaj jej jako pewnej szansy.")
    p = usable[tick].tail(260).copy()
    fig = px.line(p.reset_index(), x="Date" if p.index.name == "Date" else p.index.name or "index",
                  y="Close", title=f"{STOCKS[tick]} — ostatnie ~260 sesji")
    fig.update_layout(xaxis_title="Data", yaxis_title="Kurs zamknięcia [PLN]")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Fundamenty (ostatnie dostępne, NIE historyczne)")
    if not info:
        st.info("Dostawca nie udostępnił fundamentów. Model nie dopisuje fikcyjnych danych.")
    else:
        f1, f2, f3 = st.columns(3)
        f1.write(f"**C/Z:** {show_optional(info.get('trailingPE'))}")
        f1.write(f"**C/WK:** {show_optional(info.get('priceToBook'))}")
        f2.write(f"**Marża netto:** {show_optional(info.get('profitMargins'), is_percent=True)}")
        f2.write(f"**Wzrost przychodów:** {show_optional(info.get('revenueGrowth'), is_percent=True)}")
        f3.write(f"**Dług/kapitał:** {show_optional(info.get('debtToEquity'), unit='%')}")
        f3.write(f"**ROE:** {show_optional(info.get('returnOnEquity'), is_percent=True)}")
        st.write(f"**Cel cenowy analityków (zwykle ~12 mies., inny horyzont niż model):** "
                 f"{show_optional(info.get('targetMeanPrice'), unit=' zł')}")
        st.write(f"**Liczba opinii analityków:** {info.get('numberOfAnalystOpinions') or 'brak danych'}")
        st.write(f"**Zbiorcza opinia analityków:** {info.get('recommendationKey') or 'brak danych'}")
        st.caption("Cele analityków są pokazywane informacyjnie, nie mieszamy ich z 21-sesyjną wyceną. "
                   "Źródło: Yahoo Finance, który może mieć brakujące lub opóźnione wartości.")
    st.subheader("Czynniki zewnętrzne — korelacja, NIE przyczyna")
    if row["Korelacje"]:
        correlations = pd.DataFrame([{"Czynnik": k, "Korelacja": v}
                                     for k, v in row["Korelacje"].items()])
        st.dataframe(correlations, use_container_width=True, hide_index=True)
        st.caption("Korelacje dziennych zmian cen z ~250 sesji, z opóźnieniem wskaźników zewnętrznych.")
    else:
        st.info("Za mało wspólnych danych makro do obliczenia korelacji.")
    st.subheader("Wiadomości wyszukane automatycznie")
    st.caption("Tylko nagłówki RSS i odnośniki. To nie oznacza przeczytania pełnych artykułów ani potwierdzenia opinii ekspertów.")
    if news_url:
        st.link_button("🔎 Otwórz wyniki wyszukiwania Google News", news_url)
    if news:
        for item in news[:6]:
            st.markdown(f"- [{item['title']}]({item['url']}) — {item['source']}, {item['date']}")
    else:
        st.info("Brak aktualnych nagłówków lub chwilowy problem z kanałem RSS.")
    with st.expander("Jak liczymy wynik i jak dobry jest model?"):
        st.markdown("""
**Model 1:** Random Forest przewiduje kierunek po 21 sesjach na podstawie historii cen,
zmienności, RSI, wolumenu, trendów i opóźnionych wskaźników makro.

**Model 2:** Regresja Ridge przewiduje procentową zmianę ceny.

**Model 3:** Gradient Boosting przewiduje procentową zmianę ceny.

**Wynik 0–100:** 65% indeksu pochodzi z niekalibrowanego sygnału kierunkowego,
35% z przeskalowanej prognozy zmiany ceny. Następnie dodawane są niewielkie,
**jawnie heurystyczne** korekty za dostępne wskaźniki finansowe (do ±5 pkt)
i nagłówki (do ±2 pkt). To NIE jest matematycznie potwierdzone prawdopodobieństwo.

Makro wpływa na modele przez opóźnione zmiany SPY, USD/PLN, ropy, miedzi i VIX.
Dodatkowo widać prostą korelację cen, która nie potwierdza związku przyczynowego.

**Walidacja:** ostatnie ~65 sesji z dostępną etykietą przyszłej ceny odłożono na test,
a przed testem zastosowano ~35-dniową przerwę (embargo). Fundamenty i newsy są
użyte tylko jako aktualna nakładka, bo brak ich historycznych wersji dla uczciwego backtestu.
To oznacza, że pokazana trafność dotyczy **części ML**, a nie końcowego indeksu 0–100.
""")
        m = models[3]
        st.write({"Trafność klasyfikacji": f"{m['hit_rate']:.1%}",
                  "Częstość wzrostów w teście": f"{m['actual_up_rate']:.1%}",
                  "Brier (mniej = lepiej)": round(m["brier"], 4),
                  "MAE regresji": f"{m['mae']:.2%}",
                  "MAE naiwnego modelu": f"{m['mae_baseline']:.2%}"})
        if m["mae"] >= m["mae_baseline"]:
            st.error("Uwaga: w tym teście model NIE pobił naiwnej prognozy średniej zmiany.")
        if m["hit_rate"] <= max(m["actual_up_rate"], 1-m["actual_up_rate"]):
            st.error("Uwaga: trafność kierunku NIE pobiła bazowego zgadywania dominującej klasy.")
    st.caption("⚠️ Źródła Yahoo i Google News są przeznaczone tu do testów osobistych. "
               "Przed publikacją lub sprzedażą konieczne są odpowiednie licencje na dane, "
               "weryfikacja metodologii i ocena obowiązków regulacyjnych.")


if __name__ == "__main__":
    main()
