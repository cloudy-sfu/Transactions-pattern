select c.customer_id,
    floor(max(min(
        julianday(t.date_0) - julianday('2018-01-01'), julianday('2020-05-12') - julianday('2018-01-01')
    ), 0) / 7) as period,
    sum(t.weight * t.quality) / sum(t.weight) as quality,
    sum(t.weight * t.price) / sum(t.weight) as price,
    log(sum(t.weight) + 1) as ln_weight
from customer_scores c join transactions t on c.customer_id = t.customer_id
group by t.customer_id, period order by t.customer_id, period;
