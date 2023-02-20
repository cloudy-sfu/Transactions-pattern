select iif(score >= 43.548, iif(score >= 52.4506, iif(score >= 64.6253, 'A', 'B'), 'C'), 'D') as level,
       1.0 * count(*) / (select count(*) from customer_scores) as ratio
-- 1.0 cast integer to float
from customer_scores group by level;
