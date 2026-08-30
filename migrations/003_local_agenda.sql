-- ============================================================
-- AI Booking – Agenda locale (rimozione n8n / Google Calendar)
-- Eseguire su Supabase dopo 001_initial_schema.sql e 002_onboarding_schema.sql
-- ============================================================

-- Serve per l'exclusion constraint anti-sovrapposizione (range temporali)
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- ------------------------------------------------------------
-- 1. APPOINTMENTS – durata esplicita + sorgente + blocchi manuali
-- ------------------------------------------------------------
ALTER TABLE public.appointments
  ADD COLUMN IF NOT EXISTS duration_minutes integer NOT NULL DEFAULT 30,
  ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'whatsapp',
  ADD COLUMN IF NOT EXISTS created_by uuid REFERENCES auth.users(id) ON DELETE SET NULL;

-- source: da dove arriva la riga. 'block' = blocco manuale senza cliente/servizio.
ALTER TABLE public.appointments
  DROP CONSTRAINT IF EXISTS appointments_source_check;
ALTER TABLE public.appointments
  ADD CONSTRAINT appointments_source_check
  CHECK (source IN ('whatsapp', 'manual', 'block'));

-- status: aggiungiamo 'no_show' oltre a confirmed/cancelled/completed
ALTER TABLE public.appointments
  DROP CONSTRAINT IF EXISTS appointments_status_check;
ALTER TABLE public.appointments
  ADD CONSTRAINT appointments_status_check
  CHECK (status IN ('confirmed', 'cancelled', 'completed', 'no_show'));

-- I blocchi (source='block') non hanno cliente né servizio: rendiamo
-- esplicito che è un caso ammesso (le colonne erano già nullable).
COMMENT ON COLUMN public.appointments.duration_minutes IS
  'Durata effettiva riservata (servizio + buffer, o durata libera per i blocchi manuali)';
COMMENT ON COLUMN public.appointments.source IS
  'whatsapp = creato dal bot, manual = creato dallo staff in agenda, block = blocco orario senza cliente';

-- ------------------------------------------------------------
-- 2. Colonna generata: intervallo temporale assoluto (tstzrange)
--    Calcolato da appointment_date + appointment_time + duration_minutes,
--    nel fuso Europe/Rome per coerenza con il resto del sistema.
--    NB: se in futuro un tenant avrà un fuso diverso, valutare di
--    portare il fuso su colonna e ricalcolare qui.
-- ------------------------------------------------------------
ALTER TABLE public.appointments
  ADD COLUMN IF NOT EXISTS time_range tstzrange
  GENERATED ALWAYS AS (
    tstzrange(
      (appointment_date + appointment_time) AT TIME ZONE 'Europe/Rome',
      (appointment_date + appointment_time) AT TIME ZONE 'Europe/Rome'
        + (duration_minutes || ' minutes')::interval,
      '[)'
    )
  ) STORED;

-- ------------------------------------------------------------
-- 3. Vincolo anti-sovrapposizione: nello stesso tenant due righe
--    'confirmed' non possono avere time_range che si sovrappongono.
--    NB: volutamente NON differenziato per sede, perché il motore di
--    disponibilità (come il vecchio Google Calendar unico) tratta gli
--    impegni come un unico calendario condiviso per tutto lo studio,
--    indipendentemente dalla sede in cui si svolgono. Se in futuro un
--    tenant avrà professionisti/calendari indipendenti per sede, va
--    aggiunta una colonna "resource_id" e inclusa qui.
-- ------------------------------------------------------------
ALTER TABLE public.appointments
  DROP CONSTRAINT IF EXISTS appointments_no_overlap;
ALTER TABLE public.appointments
  ADD CONSTRAINT appointments_no_overlap
  EXCLUDE USING gist (
    tenant_id WITH =,
    time_range WITH &&
  )
  WHERE (status = 'confirmed');

CREATE INDEX IF NOT EXISTS idx_appointments_tenant_range
  ON public.appointments USING gist (tenant_id, time_range);

-- ------------------------------------------------------------
-- 4. appointment_date/time restano NOT NULL solo per source != 'block'
--    generico? In realtà anche i blocchi hanno data/ora, quindi restano
--    NOT NULL per tutti. Nessuna modifica necessaria qui.
-- ------------------------------------------------------------
