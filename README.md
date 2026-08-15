# Trading research — etap 1

Ten projekt jest **wyłącznie środowiskiem badawczym**. Nie składa zleceń i nie jest rekomendacją inwestycyjną.

## Co działa

`confluence_backtester.py` odczytuje dane OHLCV z CSV i testuje bardzo wąską hipotezę: FVG o istotnym rozmiarze (co najmniej 0,1 ATR) występuje wraz z liquidity sweepem i świecą zamykającą się w kierunku odwrócenia. Sygnał jest znany po zamknięciu świecy; wejście następuje dopiero po otwarciu następnej. To ogranicza look-ahead bias.

Wynik zawiera transakcje oraz wygładzony estymator sukcesu Beta(1,1). Nie należy go traktować jako gotowego prawdopodobieństwa dla następnego trade'u: zanim go użyjemy, musi przejść walidację out-of-sample i walk-forward.

## Format danych

Minimalny CSV (rosnąco po czasie):

```csv
timestamp,open,high,low,close,volume
2026-01-01T09:00:00Z,100,102,99,101,1234
```

## Uruchomienie

```bash
python3 confluence_backtester.py dane.csv --lookback 20 --reward-risk 2 --max-hold 20 --out wyniki.json
```

## Zanim nazwiemy to strategią

1. Ustalamy rynek, interwał, źródło danych, strefę czasową i koszty (spread, prowizja, poślizg).
2. Definiujemy FVG i liquidity sweep w sposób mechaniczny, bez ręcznej interpretacji wykresu.
3. Dzielimy dane czasowo na train / validation / final holdout, a następnie robimy walk-forward.
4. Porównujemy z prostym benchmarkiem i sprawdzamy liczbę transakcji, oczekiwaną wartość, maksymalne obsunięcie oraz stabilność parametrów.

Szczególnie ważne: FVG i liquidity sweep nie są niezależnymi dowodami — traktowanie ich jako prostego sumowania „confluence” może sztucznie zawyżyć pewność. W kolejnej iteracji zbierzemy cechy osobno i ocenimy ich wkład na danych poza próbą.
