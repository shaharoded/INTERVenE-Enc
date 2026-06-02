import transform_emr.config.model_config as mc
mc.TRAINING_SETTINGS.update({
    'sample': 50,
    'phase1_n_epochs': 1,
    'phase2_n_epochs': 1,
    'phase3_n_epochs': 1,
})
print('[Smoke] settings patched:', {k: mc.TRAINING_SETTINGS[k] for k in ('sample','phase1_n_epochs','phase2_n_epochs','phase3_n_epochs')})
import api  # runs the full pipeline at import-time
