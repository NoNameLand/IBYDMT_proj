import clip
import open_clip
from transformers import (
    AlignModel,
    AlignProcessor,
    BlipForImageTextRetrieval,
    BlipProcessor,
    FlavaModel,
    FlavaProcessor,
)

from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c


def get_clip_text_encoder(backbone, device=c.DEVICE):
    model, _ = clip.load(backbone, device=device)
    tokenize = clip.tokenize

    def encode_text(text):
        text = tokenize(text).to(device)
        return model.encode_text(text)

    return encode_text


def get_openclip_text_encoder(
    backbone, backbone_to_weights=c.OPENCLIP_WEIGHTS, device=c.DEVICE
):
    model = open_clip.create_model(
        backbone, pretrained=backbone_to_weights[backbone], device=device
    )
    tokenize = open_clip.get_tokenizer(backbone)

    def encode_text(text):
        text = tokenize(text).to(device)
        return model.encode_text(text)

    return encode_text


def get_flava_text_encoder(device=c.DEVICE):
    model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")

    def encode_text(text):
        text_inputs = processor.tokenizer(
            text, return_tensors="pt", padding="max_length", max_length=77
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        return model.get_text_features(**text_inputs)[:, 0, :]

    return encode_text


def get_align_text_encoder(device=c.DEVICE):
    model = AlignModel.from_pretrained("kakaobrain/align-base").to(device)
    processor = AlignProcessor.from_pretrained("kakaobrain/align-base")

    def encode_text(text):
        text_inputs = processor.tokenizer(
            text, return_tensors="pt", padding="max_length", max_length=77
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        return model.get_text_features(**text_inputs)

    return encode_text


def get_blip_text_encoder(device=c.DEVICE):
    model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco"
    ).to(device)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

    def encode_text(text):
        text_inputs = processor(
            text=text, return_tensors="pt", padding="max_length", max_length=77
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        question_embeds = model.text_encoder(**text_inputs)[0]
        return model.text_proj(question_embeds[:, 0, :])

    return encode_text


def get_clip_image_encoder(backbone, device=c.DEVICE):
    model, preprocess = clip.load(backbone, device=device)

    def encode_image(image):
        image = preprocess(image).unsqueeze(0).to(device)
        return model.encode_image(image)

    return encode_image


def get_open_clip_image_encoder(
    backbone, backbone_to_weights=c.OPENCLIP_WEIGHTS, device=c.DEVICE
):
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, pretrained=backbone_to_weights[backbone], device=device
    )

    def encode_image(image):
        image = preprocess(image).unsqueeze(0).to(device)
        return model.encode_image(image)

    return encode_image


def get_flava_image_encoder(device=c.DEVICE):
    model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
    processor = FlavaProcessor.from_pretrained("facebook/flava-full")

    def encode_image(image):
        image_input = processor.image_processor(images=[image], return_tensors="pt")
        image_input = {k: v.to(device) for k, v in image_input.items()}
        return model.get_image_features(**image_input)[:, 0, :]

    return encode_image


def get_align_image_encoder(device=c.DEVICE):
    model = AlignModel.from_pretrained("kakaobrain/align-base").to(device)
    processor = AlignProcessor.from_pretrained("kakaobrain/align-base")

    def encode_image(image):
        image_input = processor.image_processor(images=[image], return_tensors="pt")
        image_input = {k: v.to(device) for k, v in image_input.items()}
        return model.get_image_features(**image_input)

    return encode_image


def get_blip_image_encoder(device=c.DEVICE):
    model = BlipForImageTextRetrieval.from_pretrained(
        "Salesforce/blip-itm-base-coco"
    ).to(device)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

    def encode_image(image):
        image_inputs = processor(images=[image], return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_embeds = model.vision_model(**image_inputs)[0]
        return model.vision_proj(image_embeds[:, 0, :])

    return encode_image


def get_text_encoder(config: Config, device=c.DEVICE):
    backbone = config.data.backbone.split(":")
    if len(backbone) == 1:
        backbone.append(None)
    library, backbone = backbone

    if library == "clip":
        encode_text = get_clip_text_encoder(backbone, device=device)
    elif library == "open_clip":
        encode_text = get_openclip_text_encoder(backbone, device=device)
    elif library == "flava":
        encode_text = get_flava_text_encoder(device=device)
    elif library == "align":
        encode_text = get_align_text_encoder(device=device)
    elif library == "blip":
        encode_text = get_blip_text_encoder(device=device)
    else:
        raise NotImplementedError(f"Unknown library: {library}")
    return encode_text


def get_image_encoder(config: Config, device=c.DEVICE):
    backbone = config.data.backbone.split(":")
    if len(backbone) == 1:
        backbone.append(None)
    library, backbone = backbone

    if library == "clip":
        encode_image = get_clip_image_encoder(backbone, device=device)
    elif library == "open_clip":
        encode_image = get_open_clip_image_encoder(backbone, device=device)
    elif library == "flava":
        encode_image = get_flava_image_encoder(device=device)
    elif library == "align":
        encode_image = get_align_image_encoder(device=device)
    elif library == "blip":
        encode_image = get_blip_image_encoder(device=device)
    else:
        raise NotImplementedError(f"Unknown library: {library}")
    return encode_image
