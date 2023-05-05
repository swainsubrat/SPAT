# ----------- FGSM -----------
#python attack_main.py --model-name gtsrb_cnn_1 --ae-name cnn_128 --batch-size 512 --attack-name fgsm

# ----------- PGD ------------
#python attack_main.py --model-name gtsrb_cnn_1 --ae-name cnn_128 --batch-size 512 --attack-name pgd

# ----------- BIM ------------
#python attack_main.py --model-name gtsrb_cnn_1 --ae-name cnn_128 --batch-size 512 --attack-name bim

# ----------- CnW ------------
python attack_main.py --model-name gtsrb_cnn_1 --ae-name cnn_128 --batch-size 256 --attack-name cnw

# --------- Deepfool ---------
# python attack_main.py --model-name gtsrb_cnn_1 --ae-name cnn_128 --batch-size 256 --attack-name deepfool

# --------- Elastic ----------
python attack_main.py --model-name gtsrb_cnn_1 --ae-name cnn_128 --batch-size 256 --attack-name elastic
