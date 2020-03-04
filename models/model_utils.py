from models.humanboxes import HumanBoxes

MODEL_HumanBoxes = "HumanBoxes"

def get_model_architecture(model_name, **kwargs):
    if model_name == MODEL_HumanBoxes:
        return HumanBoxes(**kwargs)
    else:
        raise "Model {} is not valid".format(model_name)
