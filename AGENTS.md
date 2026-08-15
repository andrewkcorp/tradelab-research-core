# TradeLab — zasady współpracy agentów

## Cel i granice

TradeLab jest środowiskiem badawczym dla hipotez opartych na danych OHLCV. Nie jest systemem live tradingowym i nie może wysyłać zleceń ani przechowywać danych uwierzytelniających brokera.

## Role

- Claude Code: implementuje tylko zadanie opisane w `issues/` na osobnym branchu.
- Codex: recenzuje metodologię, kod, testy i ryzyko błędów badawczych.
- Właściciel projektu: decyduje o merge oraz o zmianach definicji strategii.

## Reguły implementacji

1. Nie zmieniaj definicji setupu, parametrów ani kryterium sukcesu bez wyraźnej aktualizacji specyfikacji i uzasadnienia.
2. Sygnał utworzony z danych świecy `t` może zostać wypełniony najwcześniej na otwarciu świecy `t+1`.
3. Przy danych OHLC, gdy stop i target zostaną dotknięte w tej samej świecy, stosuj konserwatywne założenie: najpierw stop.
4. Każda nowa reguła musi mieć test jednostkowy oraz test przypadku brzegowego.
5. Nie dodawaj zależności, usług zewnętrznych ani automatycznego execution bez akceptacji właściciela.
6. Nie commituj danych API, haseł, tokenów, pełnych danych klientów ani danych brokera.

## Git i handoff

1. Zaktualizuj `main`, utwórz branch `claude/NNN-opis` i pracuj wyłącznie na nim.
2. Zanim zgłosisz zmianę, uruchom testy oraz podaj dokładne polecenie i wynik.
3. PR ma zawierać: zakres, założenia, testy, ograniczenia i potencjalne źródła biasu.
4. Nie merguj do `main`; oczekuj na review.
