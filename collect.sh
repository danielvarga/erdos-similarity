n=5
lower_bound=15

for (( i=1 ; i<1000 ; ++i ))
do
    python main.py --n $n --random_seed $i --lower_bound $lower_bound | grep -v Seed >> collection.n$n.txt
done
