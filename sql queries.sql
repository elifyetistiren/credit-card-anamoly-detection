/*Now, let's run some basic analysis with SQL queries
For example: Count the number of fraudulent vs non-fraudulent transactions*/
SELECT Class, COUNT(*) FROM creditcard_data GROUP BY Class;

/* 

to check table info 

Option 1: Use the Access GUI (Easiest)
Open your Access database.
In the Navigation Pane, find the creditcard_data table.
Right-click the table → select Design View.*/


/* Execute a SQL query to check for NULL values in the SQLite table*/

    SELECT 
        COUNT(*) 
    FROM creditcard_data 
    WHERE Time IS NULL OR 
          V1 IS NULL OR V2 IS NULL OR V3 IS NULL OR V4 IS NULL OR V5 IS NULL OR 
          V6 IS NULL OR V7 IS NULL OR V8 IS NULL OR V9 IS NULL OR V10 IS NULL OR 
          V11 IS NULL OR V12 IS NULL OR V13 IS NULL OR V14 IS NULL OR V15 IS NULL OR 
          V16 IS NULL OR V17 IS NULL OR V18 IS NULL OR V19 IS NULL OR V20 IS NULL OR 
          V21 IS NULL OR V22 IS NULL OR V23 IS NULL OR V24 IS NULL OR V25 IS NULL OR 
          V26 IS NULL OR V27 IS NULL OR V28 IS NULL OR Amount IS NULL OR Class IS NULL;



/* Execute a query to find duplicate rows based on all columns and delete them*/

SELECT Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount, Class, COUNT(*) 
FROM creditcard_data 
GROUP BY Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount, Class
HAVING COUNT(*) > 1;


  DELETE FROM creditcard_data
    WHERE rowid NOT IN (
        SELECT MIN(rowid)
        FROM creditcard_data
        GROUP BY Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount, Class
    );
	
	
/* Execute the query to perform temporal analysis (count fraud occurrences per hour) */
	
	
SELECT 
    INT(Time / 3600) AS hour, 
    COUNT(*) AS fraud_count 
FROM creditcard_data 
WHERE Class = 1 
GROUP BY INT(Time / 3600);


/* Execute the query to calculate average, maximum transaction amount, and count by fraud vs non-fraud */


SELECT 
    Class, 
    AVG(Amount) AS avg_amt, 
    MAX(Amount) AS max_amt, 
    COUNT(*) AS cnt 
FROM creditcard_data 
GROUP BY Class;




/* # 1. Fraud Frequency by Time of Day (Class 1)*/


SELECT 
    Switch(
        hour < 6, '00:00 - 06:00',
        hour < 12, '06:00 - 12:00',
        hour < 18, '12:00 - 18:00',
        hour >= 18, '18:00 - 24:00'
    ) AS time_of_day,
    COUNT(*) AS fraud_count
FROM (
    SELECT INT(Time / 3600) AS hour
    FROM creditcard_data
    WHERE Class = 1
) AS fraud_hours
GROUP BY Switch(
        hour < 6, '00:00 - 06:00',
        hour < 12, '06:00 - 12:00',
        hour < 18, '12:00 - 18:00',
        hour >= 18, '18:00 - 24:00'
    );
	
	
	
	
/*# 8. Total Fraud Amount*/

    SELECT SUM(Amount) AS total_fraud_loss
    FROM creditcard_data
    WHERE Class = 1;
