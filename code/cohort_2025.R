length(unique(Cohort_2025$pat_enc_csn_id))

library(dplyr)

missing_rvu <- Cohort_2025 %>%
  group_by(pat_enc_csn_id, hosp_date) %>%
  summarise(
    missing_rvu_count = sum(is.na(rvu_total)),
    total_rows = n(),
    .groups = "drop"
  )

missing_rvu