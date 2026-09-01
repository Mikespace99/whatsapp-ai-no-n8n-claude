import json
from datetime import datetime, timezone
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
# SYSTEM PROMPT UNIVERSALE
# ============================================================

def _build_system_prompt(today_str: str, weekday_str: str) -> str:
    intents_list = ", ".join(f'"{i}"' for i in ALLOWED_INTENTS)

    return f"""
Sei un classificatore di intent ed estrattore di preferenze universale per un sistema di prenotazione appuntamenti via WhatsApp.

Il tuo compito principale è tradurre qualsiasi espressione temporale umana (anche vaga, implicita, progressiva o consecutiva) in una finestra temporale e oraria precisa, espressa tramite intervalli di date (date_from/date_to) e una preferenza oraria.

Analizza il messaggio corrente dell'utente considerando attentamente il contesto fornito:
- workflow attuale
- step attuale
- ultime battute della conversazione (fondamentali per capire rettifiche o messaggi consecutivi)

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
    "date_from": null o "YYYY-MM-DD",
    "date_to": null o "YYYY-MM-DD",
    "time_preference": null o "morning" o "afternoon" o "evening" o "exact",
    "exact_time": null o "HH:MM"
  }},
  "notes": null o breve nota sul ragionamento temporale effettuato
}}

INTENT AMMESSI:
{intents_list}


REGOLE LOGICHE DI TRADUZIONE TEMPORALE UNIVERSALE:

Oggi è {weekday_str} {today_str}. Usa questa data come unico perno per tutti i calcoli sul calendario.

Devi valorizzare "preferences" calcolando SEMPRE gli intervalli corretti (date_from e date_to) in base a ciò che l'utente chiede, integrando il messaggio attuale con la cronologia precedente:

1. SINGOLO GIORNO (es. "venerdì", "il 15 settembre"):
   Imposta sia date_from che date_to allo stesso identico giorno calcolato.
   - "venerdì" (se oggi è martedì 1) → date_from="2026-09-05" e date_to="2026-09-05".
   - "venerdì" (se l'utente prima ha detto "settimana prossima") → calcola il venerdì della settimana successiva.

2. PERIODI MACRO O VAGHI (es. "settimana prossima", "inizio settimana", "nel weekend", "oggi", "domani"):
   Traduci l'espressione nella sua migliore approssimazione logica di intervallo temporale (estremi inclusi):
   - "settimana prossima" → date_from = lunedì della settimana successiva, date_to = domenica della settimana successiva.
   - "inizio settimana" (nel contesto della settimana prossima) → date_from = lunedì della settimana successiva, date_to = mercoledì della settimana successiva.
   - "nel weekend" → date_from = sabato di quella settimana, date_to = domenica di quella settimana.
   - "oggi" / "domani" → calcola le rispettive date reali in formato YYYY-MM-DD.

3. FASCE ORARIE (es. "di mattina", "pomeriggio", "alle 10:30"):
   Mappa fedelmente le preferenze orarie:
   - "mattina" → time_preference = "morning"
   - "pomeriggio" → time_preference = "afternoon"
   - "sera" o "tardi" → time_preference = "evening"
   - "alle 10:30" (orario specifico) → time_preference = "exact" e exact_time = "10:30"

4. GESTIONE DEI MESSAGGI CONSECUTIVI E RETTIFICHE (FONDAMENTALE):
   Guarda le ultime battute. Se l'utente ha appena detto "settimana prossima" e nel messaggio corrente aggiunge solo "di mattina" o "inizio settimana", mantieni la finestra temporale della settimana prossima e applica il restringimento giornaliero o la fascia oraria richiesti.
   Se invece l'utente cambia radicalmente idea ("Anzi no, preferisco oggi"), cancella il filtro precedente e sposta la finestra sulla nuova richiesta.


REGOLE SULL'INTENT "change_availability":
Se il workflow attuale è "booking" e lo step attuale è "showing_slots", e l'utente rifiuta le proposte precedenti o chiede variazioni (es: "Ma per inizio settimana non c'è posto?", "Qualcosa nel pomeriggio?", "Nessuno va bene, prova domani"), l'intent è TASSATIVAMENTE "change_availability". In questo caso, ricalcola la nuova finestra temporale/oraria integrando la richiesta attuale con il contesto precedente.

Restituisci SOLO il codice JSON, senza alcun testo prima o dopo.
""".strip()


# ============================================================
# PARSE INTENT UNIVERSALE
# ============================================================

def parse_intent(
    message_text: str,
    recent_messages: list | None = None,
    current_workflow: str = "idle",
    current_step: str = "none",
    timezone_str: str = "Europe/Rome",
) -> dict:
    """
    Chiama l'AI iniettando il contesto storico e temporale.
    L'IA restituisce le date già risolte in range (date_from/to).
    Pulisce l'output per essere perfettamente digerito dal motore locale.
    """

    recent_messages = recent_messages or []

    # Configurazione Timezone del Tenant
    try:
        tz = ZoneInfo(timezone_str)
    except Exception:
        tz = timezone.utc

    now = datetime.now(tz)
    today_str = now.strftime("%Y-%m-%d")

    weekday_map = {
        0: "lunedì", 1: "martedì", 2: "mercoledì", 3: "giovedì",
        4: "venerdì", 5: "sabato", 6: "domenica"
    }
    weekday_str = weekday_map[now.weekday()]

    # Costruzione stringa Cronologia (Ultime 5 battute per il merge semantico dell'IA)
    history_text = ""
    if recent_messages:
        history_text = "Ultime battute della conversazione per comprendere il contesto:\n"
        for m in recent_messages[-5:]:
            role = "Cliente" if m.get("role") == "user" else "Assistente"
            history_text += f"- {role}: {m.get('content') or m.get('text', '')}\n"
        history_text += "\n"

    # Costruzione del payload di contesto per l'utente AI
    user_content = (
        f"{history_text}"
        f"Workflow attuale backend: {current_workflow}\n"
        f"Step attuale backend: {current_step}\n"
        f"Data/Ora di riferimento attuale del server locale: {weekday_str} {today_str} ore {now.strftime('%H:%M')}\n"
        f"Messaggio corrente da classificare ed estrarre: {message_text}"
    )

    system_prompt = _build_system_prompt(today_str, weekday_str)

    try:
        response = client.chat.completions.create(
            model=Config.AI_MODEL_INTENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,  # Bassissima temperatura per garantire massima stabilità
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        # Normalizzazione Intent
        intent = data.get("intent", INTENT_UNCLEAR)
        if intent not in ALLOWED_INTENTS:
            intent = INTENT_UNCLEAR

        # Normalizzazione Confidence
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        entities = data.get("entities") or {}
        preferences = data.get("preferences") or {}

        # Costruiamo il dizionario delle preferenze ripulendo le chiavi legacy.
        # Questo forza il motore locale a saltare i controlli su 'date' o 'period'
        # e a usare direttamente il blocco: elif prefs.get("date_from") and prefs.get("date_to"):
        clean_preferences = {
            "date_from": preferences.get("date_from"),
            "date_to": preferences.get("date_to"),
            "time_preference": preferences.get("time_preference"),
            "exact_time": preferences.get("exact_time"),
            
            # Chiavi legacy esplicitamente a None per evitare conflitti nel motore locale
            "date": None,
            "period": None,
            "weekday": None
        }

        return {
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "preferences": clean_preferences,
            "notes": data.get("notes"),
        }

    except Exception as e:
        print(f"[intent_parser_universale] Errore critico: {e}")
        return {
            "intent": INTENT_UNCLEAR,
            "confidence": 0.0,
            "entities": {},
            "preferences": {
                "date_from": None,
                "date_to": None,
                "time_preference": None,
                "exact_time": None,
                "date": None,
                "period": None,
                "weekday": None
            },
            "notes": f"Exception: {str(e)}",
        }
