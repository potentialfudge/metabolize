-- ============================================================================
-- Meta Acquisition Blender App — Database Schema
--
-- Identity is provided by Auth0 (via Streamlit's st.login()), registered in
-- Supabase as a Third-Party Auth issuer -- NOT Supabase's own auth.users.
-- That's why user_id is `text` (Auth0's `sub` claim, e.g. "auth0|abc123" or
-- "google-oauth2|456") with no foreign key to auth.users, and why RLS
-- policies check auth.jwt() ->> 'sub' rather than auth.uid().
-- ============================================================================

-- ---------------------------------------------------------------------------
-- campaigns: one row per campaign a user creates
-- ---------------------------------------------------------------------------
create table if not exists campaigns (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    name text not null,
    mode text not null check (mode in ('simple', 'advanced')),
    status text not null default 'setup' check (status in ('setup', 'active', 'completed')),
    config jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- campaign_rounds: one row per round within a campaign (round 0 = init)
-- ---------------------------------------------------------------------------
create table if not exists campaign_rounds (
    id uuid primary key default gen_random_uuid(),
    campaign_id uuid not null references campaigns(id) on delete cascade,
    round_number int not null,
    ingested boolean not null default false,
    ingested_at timestamptz,
    blender_output jsonb,
    created_at timestamptz not null default now(),
    unique (campaign_id, round_number)
);

-- ---------------------------------------------------------------------------
-- campaign_data_points: one row per experiment (param values + target)
-- ---------------------------------------------------------------------------
create table if not exists campaign_data_points (
    id uuid primary key default gen_random_uuid(),
    campaign_id uuid not null references campaigns(id) on delete cascade,
    round_number int not null,
    param_values jsonb not null,
    target_value numeric,
    created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- Indexes for common lookups
-- ---------------------------------------------------------------------------
create index if not exists idx_campaigns_user_id on campaigns(user_id);
create index if not exists idx_campaign_rounds_campaign_id on campaign_rounds(campaign_id);
create index if not exists idx_campaign_data_points_campaign_id on campaign_data_points(campaign_id);

-- ---------------------------------------------------------------------------
-- Table-level grants: allow anon (pre-login) and authenticated (post-login)
-- roles to attempt queries at all. RLS policies below still independently
-- filter which ROWS each request can see -- this just grants permission to
-- query the table in the first place.
-- ---------------------------------------------------------------------------
grant select, insert, update, delete on public.campaigns to anon, authenticated;
grant select, insert, update, delete on public.campaign_rounds to anon, authenticated;
grant select, insert, update, delete on public.campaign_data_points to anon, authenticated;

-- ---------------------------------------------------------------------------
-- Row Level Security: enable on every table
-- ---------------------------------------------------------------------------
alter table campaigns enable row level security;
alter table campaign_rounds enable row level security;
alter table campaign_data_points enable row level security;

-- ---------------------------------------------------------------------------
-- Policies: campaigns — user can only see/edit their own rows.
-- auth.jwt() ->> 'sub' reads the `sub` claim from the verified Auth0 token
-- (Supabase Third-Party Auth must have Auth0 registered as a trusted issuer
-- for auth.jwt() to see these claims).
-- ---------------------------------------------------------------------------
create policy "select own campaigns"
    on campaigns for select
    using ((auth.jwt() ->> 'sub') = user_id);

create policy "insert own campaigns"
    on campaigns for insert
    with check ((auth.jwt() ->> 'sub') = user_id);

create policy "update own campaigns"
    on campaigns for update
    using ((auth.jwt() ->> 'sub') = user_id)
    with check ((auth.jwt() ->> 'sub') = user_id);

create policy "delete own campaigns"
    on campaigns for delete
    using ((auth.jwt() ->> 'sub') = user_id);

-- ---------------------------------------------------------------------------
-- Policies: campaign_rounds — access via ownership of the parent campaign
-- ---------------------------------------------------------------------------
create policy "select own campaign rounds"
    on campaign_rounds for select
    using (
        exists (
            select 1 from campaigns
            where campaigns.id = campaign_rounds.campaign_id
            and campaigns.user_id = (auth.jwt() ->> 'sub')
        )
    );

create policy "insert own campaign rounds"
    on campaign_rounds for insert
    with check (
        exists (
            select 1 from campaigns
            where campaigns.id = campaign_rounds.campaign_id
            and campaigns.user_id = (auth.jwt() ->> 'sub')
        )
    );

create policy "update own campaign rounds"
    on campaign_rounds for update
    using (
        exists (
            select 1 from campaigns
            where campaigns.id = campaign_rounds.campaign_id
            and campaigns.user_id = (auth.jwt() ->> 'sub')
        )
    )
    with check (
        exists (
            select 1 from campaigns
            where campaigns.id = campaign_rounds.campaign_id
            and campaigns.user_id = (auth.jwt() ->> 'sub')
        )
    );

-- ---------------------------------------------------------------------------
-- Policies: campaign_data_points — same pattern, via parent campaign
-- ---------------------------------------------------------------------------
create policy "select own data points"
    on campaign_data_points for select
    using (
        exists (
            select 1 from campaigns
            where campaigns.id = campaign_data_points.campaign_id
            and campaigns.user_id = (auth.jwt() ->> 'sub')
        )
    );

create policy "insert own data points"
    on campaign_data_points for insert
    with check (
        exists (
            select 1 from campaigns
            where campaigns.id = campaign_data_points.campaign_id
            and campaigns.user_id = (auth.jwt() ->> 'sub')
        )
    );

create policy "update own data points"
    on campaign_data_points for update
    using (
        exists (
            select 1 from campaigns
            where campaigns.id = campaign_data_points.campaign_id
            and campaigns.user_id = (auth.jwt() ->> 'sub')
        )
    )
    with check (
        exists (
            select 1 from campaigns
            where campaigns.id = campaign_data_points.campaign_id
            and campaigns.user_id = (auth.jwt() ->> 'sub')
        )
    );

-- ---------------------------------------------------------------------------
-- Keep updated_at fresh on campaigns
-- ---------------------------------------------------------------------------
create or replace function set_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists trg_campaigns_updated_at on campaigns;
create trigger trg_campaigns_updated_at
    before update on campaigns
    for each row
    execute function set_updated_at();