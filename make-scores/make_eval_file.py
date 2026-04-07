import pandas as pd

# protocolo oficial EVAL
proto = pd.read_csv(
    "PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt",
    sep=" ", header=None
)
proto.columns = ["speaker","utt","env","attack","label"]

# scores da sua ResNet18v2 (EVAL!)
scores = pd.read_csv(
    "scores/ResNet-18v2/stft_EVAL_scores.txt",
    sep=" ", header=None
)
scores.columns = ["utt","score"]

# inverter score (seu modelo gera prob spoof)
scores["score"] = 1 - scores["score"]

# juntar protocolo + scores
merged = proto.merge(scores, on="utt")

# formato EXATO que o tDCF espera
final = merged[["utt","attack","label","score"]]

# salvar direto pronto
final.to_csv("CM_EVAL_RESNET.txt", sep=" ", header=False, index=False)

print("Arquivo CM_EVAL_RESNET.txt pronto para o tDCF ")