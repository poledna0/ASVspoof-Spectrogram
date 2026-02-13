import numpy as np
import argparse
from sklearn.metrics import roc_curve

def load_protocol(protocol):

    labels = {}

    with open(protocol) as f:
        for line in f:

            parts = line.strip().split()

            utt = parts[1]
            labels[utt] = parts[-1]

    return labels


def load_cm_scores(cm_file, labels):

    bona = []
    spoof = []

    with open(cm_file) as f:
        for line in f:

            utt, score = line.strip().split()
            score = float(score)

            if labels[utt] == "bonafide":
                bona.append(score)
            else:
                spoof.append(score)

    return np.array(bona), np.array(spoof)


def load_asv_scores(asv_file):

    tar = []
    non = []
    spoof = []

    with open(asv_file) as f:
        for line in f:

            parts = line.strip().split()

            score = float(parts[2])
            key = parts[1]

            if key == "target":
                tar.append(score)

            elif key == "nontarget":
                non.append(score)

            else:
                spoof.append(score)

    return np.array(tar), np.array(non), np.array(spoof)


def compute_eer(target, nontarget):

    scores = np.concatenate([target, nontarget])
    labels = np.concatenate([np.ones(len(target)),
                             np.zeros(len(nontarget))])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fnr - fpr))

    return fpr[idx], thresholds[idx]


def compute_tDCF(bona_cm,
                 spoof_cm,
                 tar_asv,
                 non_asv,
                 spoof_asv):

    # custos oficiais ASVspoof2019
    Ptar = 0.9405
    Pnon = 0.0095
    Pspoof = 0.05

    Cmiss_asv = 1
    Cfa_asv = 10
    Cmiss_cm = 1
    Cfa_cm = 10


    # threshold do ASV via EER
    Pmiss_asv, asv_threshold = compute_eer(tar_asv,
                                           non_asv)

    Pfa_asv = np.mean(non_asv >= asv_threshold)
    Pmiss_spoof_asv = np.mean(spoof_asv < asv_threshold)


    # thresholds CM
    all_scores = np.concatenate([bona_cm, spoof_cm])
    thresholds = np.sort(all_scores)

    tDCF = []

    for t in thresholds:

        Pmiss_cm = np.mean(bona_cm < t)
        Pfa_cm = np.mean(spoof_cm >= t)

        cost = (
            Cmiss_cm * Ptar * Pmiss_cm +
            Cfa_cm * Pspoof * Pfa_cm * (1 - Pmiss_spoof_asv) +
            Cmiss_asv * Ptar * Pmiss_asv +
            Cfa_asv * Pnon * Pfa_asv
        )

        tDCF.append(cost)

    tDCF = np.array(tDCF)

    # normalização
    default_cost = min(
        Cmiss_asv * Ptar + Cfa_asv * Pnon,
        Cmiss_cm * Ptar + Cfa_cm * Pspoof
    )

    tDCF_norm = tDCF / default_cost

    return np.min(tDCF_norm)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cm_scores", required=True) # meu qn vou gerar
    parser.add_argument("--asv_scores", required=True) # pm_scores
    parser.add_argument("--protocol", required=True) #pm_cm_protocol

    args = parser.parse_args()

    print("\nLoading protocol...")
    labels = load_protocol(args.protocol)

    print("Loading CM scores...")
    bona_cm, spoof_cm = load_cm_scores(args.cm_scores,
                                       labels)

    print("Loading ASV scores...")
    tar_asv, non_asv, spoof_asv = load_asv_scores(
        args.asv_scores
    )

    print("\nCalculating t-DCF...\n")

    tdcf = compute_tDCF(
        bona_cm,
        spoof_cm,
        tar_asv,
        non_asv,
        spoof_asv
    )

    print(f"min t-DCF = {tdcf:.5f}\n")


if __name__ == "__main__":
    main()
