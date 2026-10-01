# One-time export of nflfastR's spread-aware win probability model (MIT,
# fastrmodels 1.0.2) to xgboost JSON for backend/model/ingame_nflfastr.py.
# Run from the repository root: Rscript ops/export_nflfastr_wp.R
model <- xgboost::xgb.Booster.complete(fastrmodels::wp_model_spread)
path <- "backend/data/processed/ingame/nflfastr_wp_spread.json"
dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
invisible(xgboost::xgb.save(model, path))
