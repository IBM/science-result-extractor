#jbsub -c 1+1 -mem 15g -err err -out out python3 run_classifier.py \
# --do_train=true \
# --do_eval=false \
# --do_predict=false \
# --data_dir=./data/is_new_prediction/fold1/ \
# --output_dir=./tmp/new_fold1_output/




#jbsub -queue x86_6h -c 1+1 -mem 15g -err err -out out python3 run_classifier.py \
# --do_train=false \
# --do_eval=false \
# --do_predict=true \
# --data_dir=./data/switchboard/ \
# --output_dir=./tmp/switchboard_output/ \
# --init_checkpoint=./tmp/switchboard_output/model.ckpt-2280

jbsub -queue x86_6h -c 1+1 -mem 15g -err err -out out python3 run_classifier_sci.py \
 --do_train=false \
 --do_eval=false \
 --do_predict=true \
 --data_dir=./data/sciexp3/ \
 --output_dir=./tmp/sciexp3/ \
 --init_checkpoint=./tmp/sciexp3/model.ckpt-27128
