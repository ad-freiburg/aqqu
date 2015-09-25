These are the various final outputs of our system on the different runs in our evalution of the paper.

You can compute results using the evaluate.py script, e.g.,:

python evaluate.py results/final_wq_evaluation_output.txt

should output:

> Number of questions: 2032
> Average recall over questions: 0.604353752656
> Average precision over questions: 0.496435505027
> Average f1 over questions: 0.494262559655
> Accuracy over questions: 0.369094488189
> F1 of average recall and average precision: 0.545104629829

Files in the same format will be produced when running the evaluation using the
learner module. See QUICKSTART.md for details.
