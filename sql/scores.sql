select *
from customer_scores
where customer_id in (
    select distinct(c.customer_id)
    from customer_scores c join transactions t on c.customer_id = t.customer_id
    )
order by customer_id;
