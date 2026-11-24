"""Select the legacy I-JEPA encoder or an explicitly named external baseline."""


def init_probe_encoder(
    device,
    model_name,
    patch_size,
    crop_size,
    use_gradient_checkpointing=False,
    pretrained=True,
    meta_args=None,
    data_args=None,
    mask_args=None,
):
    """Initialize a probe encoder without changing legacy I-JEPA defaults."""
    from src.models.external_encoders import is_external_encoder

    if is_external_encoder(model_name):
        from src.models.external_encoders import build_external_encoder

        return build_external_encoder(
            model_name=model_name,
            crop_size=crop_size,
            pretrained=pretrained,
            device=device,
            meta_args=meta_args,
            data_args=data_args,
            mask_args=mask_args,
        )

    # Importing the historical implementation and constructing its unused
    # predictor are intentionally unchanged for I-JEPA checkpoint compatibility.
    from src.helper import init_model

    encoder, _ = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    return encoder
