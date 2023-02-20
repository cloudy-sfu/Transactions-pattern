select ('weight') as name, count(*) as count from transactions where weight is null
union all
select ('price') as name, count(*) as count from transactions where price is null
union all
select ('quality') as name, count(*) as count from transactions where quality is null
;