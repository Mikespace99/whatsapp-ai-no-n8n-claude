"""
AI#1 – Intent + Entities + Preferences extractor.
Restituisce sempre un JSON strutturato.
"""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from openai import OpenAI

from app.config import Config
from app.constants import (
    INTENT_GREETING,
    INTENT_BOOK,
    INTENT_RESCHEDULE,
    INTENT_CANCEL,
    INTENT_INFO,
    INTENT_SLOT_SELECTION,
    INTENT_CHANGE_AVAILABILITY,
    INTENT_CONFIRM,
    INTENT_AFFIRM,
    INTENT_DENY,
    INTENT_REQUEST_HUMAN,
    INTENT_ABANDON,
    INTENT_UNCLEAR,
)

client = OpenAI(api_key=Config.OPENAI_API_KEY)


# ============================================================
# INTENT AMMESSI
# ============================================================

ALLOWED_INTENTS = [
    INTENT_GREETING,
    INTENT_BOOK,
    INTENT_RESCHEDULE,
    INTENT_CANCEL,
    INTENT_INFO,
    INTENT_SLOT_SELECTION,
    INTENT_CHANGE_AVAILABILITY,
    INTENT_CONFIRM,
    INTENT_AFFIRM,
    INTENT_DENY,
    INTENT_REQUEST_HUMAN,
    INTENT_ABANDON,
    INTENT_UNCLEAR,
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

def _build_system_prompt(today_str: str, weekday_str: str) -> str:
    intents_list = ", ".join(f'"{i}"' for i in ALLOWED_INTENTS)

    return f"""
Sei un classificatore di intent per un sistema di prenotazione
appuntamenti via WhatsApp.

Analizza il messaggio dell'utente considerando anche:
- workflow attuale
- step attuale
- ultime battute della conversazione

Restituisci SOLO un JSON valido con questa struttura:

{{
  "intent": "uno dei valori ammessi",
  "confidence": 0.0-1.0,
  "entities": {{
    "service": null o stringa,
    "person_name": null o stringa,
    "slot_number": null o intero (1, 2, 3...),
    "selected_time": null o stringa,
    "info_type": null o "parking" o "price" o "address" o "hours" o "other"
  }},
  "preferences": {{
    "date": null o "YYYY-MM-DD",
    "date_from": null o "YYYY-MM-DD",
    "date_to": null o "YYYY-MM-DD",
    "period": null o "today" o "tomorrow" o "this_week" o "next_week",
    "weekday": null o "monday" o "tuesday" o "wednesday" o "thursday" o "friday" o "saturday" o "sunday",
    "time_preference": null o "morning" o "afternoon" o "evening" o "exact",
    "exact_time": null o "HH:MM"
  }},
  "notes": null o breve nota
}}

INTENT AMMESSI (usa ESATTAMENTE queste stringhe):
{intents_list}


REGOLE INTENT:

- Saluto
  → "greeting"

- Vuole prenotare un appuntamento
  → "book_appointment"

- Vuole spostare un appuntamento già esistente
  → "reschedule_appointment"

- Vuole cancellare un appuntamento già esistente
  → "cancel_appointment"

- Chiede informazioni
  (prezzo, parcheggio, orari, indirizzo...)
  → "get_info"
  e popola entities.info_type di conseguenza

- Sceglie uno degli slot attualmente mostrati
  ("il secondo", "numero 2", "alle 10:30")
  → "slot_selection"

- Conferma una scelta o una richiesta
  ("sì", "ok", "va bene", "confermo")
  → "confirm" oppure "affirm"

- Rifiuta una specifica scelta che gli viene chiesto di confermare
  ("no", "non va bene")
  → "deny"

- Se il workflow è "booking" e lo step è "showing_slots",
  l'utente rifiuta gli slot attualmente mostrati e vuole
  una nuova ricerca di disponibilità:
  ("nessuno va bene",
   "nessuno di questi",
   "prova domani",
   "hai qualcosa venerdì?",
   "questi orari non vanno bene",
   "vediamo settimana prossima",
   "domani hai qualcosa?")
  → "change_availability"

IMPORTANTE:

"change_availability" NON significa cancellare o spostare
un appuntamento già esistente.

Significa che l'utente sta effettuando una prenotazione,
ha ricevuto degli slot, non li vuole e desidera cambiare
i criteri della ricerca di disponibilità.

Quando riconosci "change_availability", estrai tutte le
nuove preferenze temporali presenti nel messaggio.

Esempi:

"Nessuno va bene, prova domani"
→ intent = "change_availability"
→ preferences.period = "tomorrow"

"Prova venerdì pomeriggio"
→ intent = "change_availability"
→ preferences.weekday = "friday"
→ preferences.time_preference = "afternoon"

"Hai qualcosa settimana prossima?"
→ intent = "change_availability"
→ preferences.period = "next_week"

"Nessuno di questi va bene"
→ intent = "change_availability"
→ nessuna nuova preferenza temporale


IMPORTANTE SULLA DISTINZIONE:

"il secondo"
→ "slot_selection"

"alle 16:00"
→ "slot_selection"

"no"
mentre viene chiesto di confermare uno specifico slot
→ "deny"

"nessuno va bene, prova domani"
→ "change_availability"

"questi orari non vanno bene, hai qualcosa venerdì?"
→ "change_availability"


DATA ODIERNA:

Oggi è {weekday_str} {today_str}.

REGOLE SU DATE E GIORNI DELLA SETTIMANA — IMPORTANTE:

NON calcolare MAI tu la data quando l'utente fa riferimento a un
giorno della settimana ("venerdì", "lunedì prossimo", "mercoledì
della prossima settimana", ecc.). In questi casi usa SOLO:
- preferences.weekday → il giorno della settimana menzionato
  ("monday".."sunday")
- preferences.period → "next_week" se l'utente dice esplicitamente
  "prossima settimana"/"settimana prossima", "this_week" se dice
  "questa settimana", altrimenti null

Il calcolo della data esatta a partire da weekday+period viene fatto
da un programma, non da te: è più affidabile e non deve mai sbagliare.

Usa invece preferences.date SOLO quando l'utente esprime una data
assoluta ed esplicita (es. "il 15 settembre", "15/09", "2026-09-20"):
in quel caso calcola tu la data in formato YYYY-MM-DD.

Per "oggi" e "domani" usa preferences.period = "today" / "tomorrow"
(non calcolare tu la data anche in questo caso).

Esempi:

"Vorrei un appuntamento mercoledì della prossima settimana"
→ preferences.weekday = "wednesday"
→ preferences.period = "next_week"
→ (NON preferences.date)

"Hai qualcosa venerdì?"
→ preferences.weekday = "friday"
→ (NON preferences.date)

"Il 20 settembre va bene?"
→ preferences.date = "2026-09-20" (data assoluta, qui sì che la calcoli tu)

Le preferenze di data/ora sono criteri per la ricerca
di disponibilità e devono essere estratte fedelmente.
Se l'utente esprime SOLO una fascia oraria (mattina/pomeriggio/sera)
senza indicare alcun giorno, estrai solo preferences.time_preference
e lascia data/weekday/period a null: è una preferenza valida da sola.

Restituisci SOLO il JSON, nient'altro.
""".strip()


# ============================================================
# RISOLUZIONE DETERMINISTICA weekday + period → date
# ============================================================
# L'AI estrae SOLO etichette (weekday, period): il calcolo della data
# esatta lo fa questo codice, non il modello. Vedi discussione con
# l'utente: lasciare l'aritmetica delle date a un LLM è intermittente
# e difficile da testare (es. "mercoledì della prossima settimana"
# calcolato a volte correttamente, a volte come "lunedì prossimo").

_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _resolve_weekday_preference(preferences: dict, now: datetime) -> dict:
    """
    Se preferences contiene un 'weekday' valido, lo traduce in una
    'date' concreta (YYYY-MM-DD) e rimuove weekday/period, così tutto
    il resto del sistema (decision.py, booking/engine.py) continua a
    ricevere preferences.date già risolta, esattamente come prima.

    Regole (decise insieme):
    - weekday + period="next_week" → quel giorno nella settimana di
      calendario (lun-dom) successiva a quella corrente
    - weekday + period="this_week" → quel giorno nella settimana di
      calendario corrente (se già passato, il motore di disponibilità
      restituirà "nessuno slot" e scatta il fallback esistente:
      comportamento accettabile, nessuna gestione speciale)
    - weekday da solo (nessun period) → se il giorno non è ancora
      passato questa settimana lo uso, altrimenti quello della
      settimana successiva
    """
    weekday = preferences.get("weekday")
    if not weekday or weekday not in _WEEKDAY_INDEX:
        return preferences

    target_idx = _WEEKDAY_INDEX[weekday]
    today_idx = now.weekday()  # Monday=0 .. Sunday=6, coerente con _WEEKDAY_INDEX
    period = preferences.get("period")

    if period == "next_week":
        days_to_next_monday = (7 - today_idx) % 7 or 7
        next_monday = now.date() + timedelta(days=days_to_next_monday)
        resolved = next_monday + timedelta(days=target_idx)
    elif period == "this_week":
        this_monday = now.date() - timedelta(days=today_idx)
        resolved = this_monday + timedelta(days=target_idx)
    else:
        # Nessun modificatore di settimana: prossima occorrenza utile
        # (oggi stesso se coincide, altrimenti il primo giorno buono
        # da qui in avanti, anche nella settimana successiva)
        delta = (target_idx - today_idx) % 7
        resolved = now.date() + timedelta(days=delta)

    preferences = dict(preferences)
    preferences["date"] = resolved.isoformat()
    preferences.pop("weekday", None)
    preferences.pop("period", None)
    return preferences


# ============================================================
# PARSE INTENT
# ============================================================

def parse_intent(
    message_text: str,
    recent_messages: list | None = None,
    current_workflow: str = "idle",
    current_step: str = "none",
    timezone_str: str = "Europe/Rome",
) -> dict:
    """
    Chiama l'AI e restituisce il dict strutturato.

    In caso di errore restituisce:
        intent = unclear
        confidence = 0.0
    """

    recent_messages = recent_messages or []

    # --------------------------------------------------------
    # Data corrente nel timezone del tenant
    # --------------------------------------------------------

    try:
        tz = ZoneInfo(timezone_str)
    except Exception:
        tz = timezone.utc

    now = datetime.now(tz)

    today_str = now.strftime("%Y-%m-%d")

    weekday_map = {
        0: "lunedì",
        1: "martedì",
        2: "mercoledì",
        3: "giovedì",
        4: "venerdì",
        5: "sabato",
        6: "domenica",
    }

    weekday_str = weekday_map[now.weekday()]

    # --------------------------------------------------------
    # Contesto storico
    # --------------------------------------------------------

    history_text = ""

    if recent_messages:
        history_text = "Ultime battute della conversazione:\n"

        for m in recent_messages[-4:]:
            role = (
                "Cliente"
                if m.get("role") == "user"
                else "Assistente"
            )

            history_text += (
                f"{role}: {m.get('content')}\n"
            )

        history_text += "\n"

    # --------------------------------------------------------
    # Messaggio inviato al modello
    # --------------------------------------------------------

    user_content = (
        f"{history_text}"
        f"Workflow attuale: {current_workflow}\n"
        f"Step attuale: {current_step}\n"
        f"Data di oggi: {weekday_str} {today_str}\n"
        f"Messaggio corrente del cliente: {message_text}"
    )

    system_prompt = _build_system_prompt(
        today_str,
        weekday_str,
    )

    # --------------------------------------------------------
    # Chiamata AI
    # --------------------------------------------------------

    try:
        response = client.chat.completions.create(
            model=Config.AI_MODEL_INTENT,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content

        data = json.loads(raw)

        # ----------------------------------------------------
        # Normalizzazione intent
        # ----------------------------------------------------

        intent = data.get(
            "intent",
            INTENT_UNCLEAR,
        )

        if intent not in ALLOWED_INTENTS:
            intent = INTENT_UNCLEAR

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        try:
            confidence = float(
                data.get("confidence", 0.5)
            )
        except (TypeError, ValueError):
            confidence = 0.5

        # ----------------------------------------------------
        # Entities / Preferences
        # ----------------------------------------------------

        entities = data.get("entities") or {}
        preferences = data.get("preferences") or {}
        preferences = _resolve_weekday_preference(preferences, now)

        return {
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "preferences": preferences,
            "notes": data.get("notes"),
        }

    except Exception as e:
        print(
            f"[intent_parser] Errore: {e}"
        )

        return {
            "intent": INTENT_UNCLEAR,
            "confidence": 0.0,
            "entities": {},
            "preferences": {},
            "notes": str(e),
        }
