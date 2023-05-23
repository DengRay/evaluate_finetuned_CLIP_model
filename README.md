# evaluate_finetuned_CLIP_model
Infimind project to access the fine-tuned CLIP model and explore how to improve

## show the modle structure
- in order to understand the detail of the CLIP model
- running the show_structure.py 
- you should download the checkout point first

## show the training loss and other curves
- running the plt_log.py to draw the picture of your log file,and analyze the training process

## zero-shot classification task
- we conduct zero-shot task on COCO dataset and a fine-grained color dataset
- COCO test data is available at:[https://drive.google.com/file/d/1DUiz95dA7blgTjYu1mOvBmM1Ism9vlj4/view?usp=share_link]
1. generate the test data for zeroshot,running generate_coco.py/COCO_txt.py, you can check the txt file to confirm the specific classes
2. run the script zeroshot_eval.sh to get accuracy result of the classifition
3. the zeroshot_evaluation.py will generate the confusion matrix
4. the zeroshot_evaluation_2.py will generate the T-SNE projection to get the classification cluster







