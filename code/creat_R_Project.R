# 1.--------------------------------------------------
# create .Rproj, R, .Rproj.user, .gitignore files
library(usethis)
usethis::create_project("/shares/nfs/kernds/k00014_delirium_ICU_admissions/delirium_prediction_LSTM/Delirium-LSTM")
# ----------------------------------------------------

# 2.--------------------------------------------------
# initiate renv for project specific environment
renv::init()
# ----------------------------------------------------


# 3.--------------------------------------------------
# git and github configuration
usethis::use_git()
# -------------(run in terminal)
usethis::use_github()
# usethis::create_github_token()
# usethis::edit_r_environ()
# ----------------------------------------------------

# 4.--------------------------------------------------
# create README.md
usethis::use_readme_md()
# ----------------------------------------------------