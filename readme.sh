
bash collect.sh
cat collection.n5.txt | grep -v "=" | cut -f2 | sort | uniq | python verify_streak_rule.py
