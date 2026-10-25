-- Record user-authorized private use without claiming an upstream license.
ALTER TABLE dataset_sources DROP CONSTRAINT dataset_sources_rights_status_check;
ALTER TABLE dataset_sources ADD CONSTRAINT dataset_sources_rights_status_check
    CHECK (rights_status IN ('unresolved', 'permitted', 'personal_use', 'restricted', 'authored'));
ALTER TABLE dataset_sources ADD CONSTRAINT dataset_sources_personal_use_evidence_check
    CHECK (rights_status <> 'personal_use' OR
        (rights_evidence IS NOT NULL AND btrim(rights_evidence) <> ''));
