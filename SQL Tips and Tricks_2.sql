--most basic query

select columnA, columnB from table1 where columnC = 'abc' order by columnA

--sort and select distinct

select distinct isnull(columnA, 'None') as columnTitle, columnB from dbo.tbl.table1

select top (100) columnA, columnB from dbo.tbl.table1 order by columnA desc;

select columnA, columnB from dbo.tbl.table1 order by columnA desc offset 0 rows fetch next 10 rows only;

--filtering and using predicates

select columnA, columnB from dbo.tbl.table1 where columnC = 6; -- <> not equal

select top 10 columnA, columnB from dbo.tbl.table1 where columnD like 'AJ%'; --AJxxxxx wildcards
select top 10 columnA, columnB from dbo.tbl.table1 where columnD like '%AJ%'; -- xxxxxAJxxxxx wildcards

select columnA, columnB from dbo.tbl.table1 where columnD like 'AJ-_[0-9][0-9]_-[0-9][0-9]' -- _underscore is any character wildcards

select columnA, columnB from dbo.tbl.table1 where columnDate is not NULL

select columnA, columnB from dbo.tbl.table1 where columnDate between '2016/1/1' and '2016/12/31'

select columnA,columnB from dbo.tbl.table1 where columnList in (5,6,7) order by columnList

select columnA,columnB from dbo.tbl.table1 where columnList in (5,6,7) and colimnDate is noy NULL order by columnList

--joins

select A.columnA, A.columnB, B.columnC, C.columnD
from dbo.tbl.table1 as A 
join dbo.tbl.table2 as B on A.columnPK = B.columnFK -- for left outer join use 'LEFT OUTER JOIN'
join dbo.tbl.table3 as C on A.columnFK = C.columnPK -- cartesian joins use 'CROSS JOIN'
where A.columnX < C.columnX
order by A.columnA desc

--self join
select emp.name as employee , man.name as manager
from HR.employee as emp left outer join HR.employee as man
on emp.managerID = man.employeeID

-- unions
select columnA, columnB from table1 
union -- use UNION ALL for faster performance and if you do NOT want distinct to be applied by default
select columnA, columnB from table2

select columnA, columnB, 'table1' as Type from table1 
union 
select columnA, columnB, 'table2' as Type from table2

-- intersect ... find overlapping records

select columnA, columnB from table1 
intersect 
select columnA, columnB from table2

-- except ... usecase below is give me a list of customers who are not employees

select firstname, lastname from tbl.customers
except
select firstname, lastname from tbl.employees

------------------------------------------------------------------------------				
--																			--
--																			--
--                      Create update and delete				            --
--																			--
--																			--
------------------------------------------------------------------------------

-- create
CREATE TABLE schemaname.tblSales
(
	OrderID int IDENTITY (1000,1) PRIMARY KEY NOT NULL, --identity sets a seed and increments it. Best used with PK.
	OrderDate datetime NOT NULL DEFAULT GETDATE(), -- defaults to current date if datetime is not specified during insertion
	SalesPerson nvarchar(256) NOT NULL, -- nvarchar stores unicode whereas varchar stores ASCII. nvarchar 2 bytes/character vs 1 byte for varchar
	CustomerID int NOT NULL REFERENCES schemaname.tblcustomer -- REFERENCES ensures referential integrity
	phonenumber varchar(25) NOT NULL,
	orderamount int NOT NULL
)

-- insert data manually

INSERT schemaname.tblSales (OrderDate, SalesPerson, CustomerID, phonenumber, orderamount) -- the column names can be avoided, if you know the table schema
VALUES
('2015-01-03T12:30:30','John Doe',123,'212-999-9999',1000),
(DEFAULT, 'Jane Doe', 234,'212-111-1111',2000);

-- insert data from a query
INSERT schemaname.tblSales (OrderDate, SalesPerson, CustomerID, phonenumber, orderamount)
Select OrderDate, 'John Doe', CustomerID, phonenumber, orderamount -- assumes all data belongs to 'John doe'
from schemaname2.tblSales 
where SalespersonID = 345

-- overriding identity ... for example if you needed to re-enter the order details of a particular order number

SET IDENTITY_INSERT schemaname.tblSales ON

INSERT schemaname.tblSales (OrderID, OrderDate, SalesPerson, CustomerID, phonenumber, orderamount)
VALUES
(1200, DEFAULT, 'John Doe', '123','212-111-1111',5000);

SET IDENTITY_INSERT schemaname.tblSales OFF

-- update table
UPDATE schemaname.tblSales
SET SalesPerson = 'John Doe Jr'
where SalesPerson = 'John Doe'

-- update multiple columns in a table
UPDATE schemaname.tblSales
SET CustomerID = 456, phonenumber = '212-999-9999'
where CustomerID = 123

-- update from a query
UPDATE schemaname.tblSales
SET CustomerID = c.CustomerID, phonenumber = c.phonenumber
FROM schemaname2.tblSales as c
where c.CustomerID = schemaname.tblSales.CustomerID

-- delete data
DELETE FROM schemaname.tblSales
where OrderDate < DATEADD(dd,-20,GETDATE()) -- delete orders older than 20 days

-- truncate an entire table
TRUNCATE TABLE schemaname.tblSales -- empties the table .. does NOT drop the table.

------------------------------------------------------------------------------				
--																			--
--																			--
--                      examples of some functions within T-SQL             --
--																			--
--																			--
------------------------------------------------------------------------------


-- date functions
select year(columnDate) as Year, datename(mm, columnDate) as Month, day(columnDate) as Day, datename(dw, columnDate) as weekday, columnA
from table1
order by columnDate

select datediff(yy, columnDate, GETDATE()) as YearsDifference, columnA
from table1
order by columnDate

-- simple string functions
select upper(Name) as ProductName -- or lower()
from table1

select concat(firstname,' ',lastname) as FullName from table1

select Name, ProductName, left(ProductNumber, 2) as ProdType -- or right()
from table1

-- complex string functions
select Name, ProductName, left(ProductNumber,2) as ProdType, 
	substring(ProductNumber,CHARINDEX('-', ProductNumber)+1,4) as ModelCode 
	/* 	CHARINDEX('search for this', 'in this column') returns index information
		substring('in this column', 'starting from this point','return 4 characters') 
	*/
	substring(ProductNumber, len(ProductNumber) - charindex('_',Reverse(Right(ProductNumber,3))) +2,2) --'FR-R2845-58'
	/*above statement. Reverse('abc') = cba
	substring(ProductNumber, [11-3+2],2) 
	*/
	
-- IIF function
select ProductName, IIF(ProductID in (5,6,7),'Bike','Other') as ProductCategory --IIF = if else
from table1

-- Ranking functions
select top (3) productID, Name, ListPrice,
	Rank() Over(order by listPrice desc) as RankByPrice
from table1
order by RankByPrice

select Category, Product, ListPrice, 
	Rank() Over(PARTITION By Category Order by ListPrice desc) as RankByPrice --ranks products within category by partitioning at category level
from table1
order by RankByPrice

-- Aggregate functions
count(*) -- returns count of rows
count(distinct ColumnA) -- returns distinct values within columnA
Avg(listprice) -- average
Max(listprice), Min(listprice) -- max and min
Sum(Revenue) -- sums revenue across all rows

select S.Salesperson, ISNULL(Sum(O.Sales),0.00) as SalesRevenue
from SalesTable S left join SalesDetailTable O on S.SalespersonID = O.SalespersonID
group by S.Salesperson
order by SalesRevenue

select S.Salesperson, concat(O.firstName,' ',O.lastName) as Customer, ISNULL(Sum(O.Sales),0.00) as SalesRevenue
from SalesTable S left join SalesDetailTable O on S.SalespersonID = O.SalespersonID
group by S.Salesperson, concat(O.firstName,' ',O.lastName) -- you cannot use the 'customer alias here because 'group by' is executed before select
order by SalesRevenue, Customer

-- where versus having
-- 'where' clause is processed BEFORE Group by ... however 'having' is processed AFTER group by

select S.Salesperson, Sum(O.Sales) as SalesRevenue
from SalesTable S left join SalesDetailTable O on S.SalespersonID = O.SalespersonID
where year(OrderDate) = 2014
group by S.Salesperson
having Sum(O.Sales) > 100
order by SalesRevenue

------------------------------------------------------------------------------				
--																			--
--																			--
--								sub queries 					            --
--																			--
--																			--
------------------------------------------------------------------------------

select * from table1
where columnA > 
(select Avg(ColumnB) from table1)

-- in the next example we see how to extract latest order detail for each customer

select customerID, orderNumber, orderDate
from salesOrderTable as SO1
where orderDate > 
(select max(orderDate) from salesOrderTable as SO2 where SO1.customerID = SO2.customerID)
order by customerID

-- complex subquery without APPLY. 

SELECT 
        CASE [Model] 
            WHEN 'Mountain-100' THEN 'M200' 
            WHEN 'Road-150' THEN 'R250' 
            WHEN 'Road-650' THEN 'R750' 
            WHEN 'Touring-1000' THEN 'T1000' 
            ELSE Left([Model], 1) + Right([Model], 3) 
        END + ' ' + [Region] AS [ModelRegion] 
        ,(Convert(Integer, [CalendarYear]) * 100) + Convert(Integer, [Month]) AS [TimeIndex] 
        ,Sum([Quantity]) AS [Quantity] 
        ,Sum([Amount]) AS [Amount]
		,CalendarYear
		,[Month]
		,[dbo].[udfBuildISO8601Date] ([CalendarYear], [Month], 25)
		as ReportingDate
    FROM 
        [dbo].[vDMPrep] 
    WHERE 
        [Model] IN ('Mountain-100', 'Mountain-200', 'Road-150', 'Road-250', 
            'Road-650', 'Road-750', 'Touring-1000') 
    GROUP BY 
        CASE [Model] 
            WHEN 'Mountain-100' THEN 'M200' 
            WHEN 'Road-150' THEN 'R250' 
            WHEN 'Road-650' THEN 'R750' 
            WHEN 'Touring-1000' THEN 'T1000' 
            ELSE Left(Model,1) + Right(Model,3) 
        END + ' ' + [Region] 
        ,(Convert(Integer, [CalendarYear]) * 100) + Convert(Integer, [Month])
		,CalendarYear
		,[Month]
		,[dbo].[udfBuildISO8601Date] ([CalendarYear], [Month], 25);

-- now using APPLY. With APPLY we don't have to keep rewriting subquery code.

    SELECT
         [ModelRegion]
        ,[TimeIndex]
        ,Sum([Quantity]) AS [Quantity]
        ,Sum([Amount]) AS [Amount]
        ,[CalendarYear]
        ,[Month]
        ,[ReportingDate]
    FROM
        [dbo].[vDMPrep]
    CROSS APPLY
    (
        SELECT
               CASE [Model]
                   WHEN 'Mountain-100' THEN 'M200'
                   WHEN 'Road-150' THEN 'R250'
                   WHEN 'Road-650' THEN 'R750'
                   WHEN 'Touring-1000' THEN 'T1000'
                   ELSE Left([Model], 1) + Right([Model], 3)
               END + ' ' + [Region] AS [ModelRegion]
             , (Convert(Integer, [CalendarYear]) * 100) + Convert(Integer, [Month]) AS [TimeIndex]
             , [dbo].[udfBuildISO8601Date] ([CalendarYear], [Month], 25) AS ReportingDate
    ) _
    WHERE
        [Model] IN ('Mountain-100', 'Mountain-200', 'Road-150', 'Road-250',
            'Road-650', 'Road-750', 'Touring-1000')
    GROUP BY
         [ModelRegion]
        ,[TimeIndex]
        ,[CalendarYear]
        ,[Month]
        ,[ReportingDate]


------------------------------------------------------------------------------				
--																			--
--																			--
--								table structures				            --
--																			--
--																			--
------------------------------------------------------------------------------

-- VIEWS. 

CREATE VIEW salesdb.vwSalesOrders -- create view 'vwSalesOrders' in the salesdb schema
AS
select S.Salesperson, ISNULL(Sum(O.Sales),0.00) as SalesRevenue, O.CustomerName
from SalesTable S left join SalesDetailTable O on S.SalespersonID = O.SalespersonID
group by S.Salesperson, O.CustomerName
order by SalesRevenue

-- now you can query the view directly as a table

-- TEMP TABLES. 
-- temp tables are not persisted in the database. the table is dropped when user session is terminated.
-- created in tempdb. maintained by SQL server system. 
-- tempdb is not ideal. Temp tables can cause recompilations.
-- you can INSERT INTO a temp tables


Create TABLE #SalesOrders -- temp tables are created with prefix #
(
select S.Salesperson, ISNULL(Sum(O.Sales),0.00) as SalesRevenue, O.CustomerName
from SalesTable S left join SalesDetailTable O on S.SalespersonID = O.SalespersonID
group by S.Salesperson, O.CustomerName
order by SalesRevenue
)
-- TABLE VARIABLES
-- useful in place of temp tables
-- use on small databases
-- works only within the batch of executed statement ... its more temporary than a temp table.

DECLARE @SalesOrders as TABLE 
(
select S.Salesperson, ISNULL(Sum(O.Sales),0.00) as SalesRevenue, O.CustomerName
from SalesTable S left join SalesDetailTable O on S.SalespersonID = O.SalespersonID
group by S.Salesperson, O.CustomerName
order by SalesRevenue
)

-- TABLE VALUED FUNCTION

CREATE FUNCTION fn_OrderedItems (@OrderID as INTEGER) -- here you can enter OrderID as a parameter
RETURNS TABLE
AS
RETURN
(Select productID, Units, ProductPrice
from Sales.OrderDetails
where OrderID = @OrderID)

-- Table valued function and table variables together

SET NOCOUNT ON -- useful when inserting so that we suppress insert messages, and improve performance

CREATE FUNCTION fn_OrderedItems (@OrderID as int)
RETURNS @OrderedItems TABLE (
	productID int PRIMARY KEY NOT NULL,
	Units int NULL
	ProductPrice int NOT NULL
	OrderDate datetime NOT NULL
	)
AS 
BEGIN
	INSERT INTO @OrderedItems (productID, Units, ProductPrice, OrderDate)
	SELECT productID, Units, ProductPrice, OrderDate
	from Sales.OrderDetails
	where OrderID = @OrderID
	
	RETURN;
END;

-- DERIVED TABLES
-- Use CTEs instead

Select Category, count(ProductID) as products
from (
	select p.productID, p.name as prod, c.Name as category
	from salestbl as p join prodtbl as c on p.pid = c.pid
	) as ProdCats --ProdCats here is a derived table
group by Category
order by Category

-- COMMON TABLE EXPRESSION (CTE)
-- similar to derived tables but unlike derived tables can support multiple references and recursion

WITH MyCTETable(ColumnA, ColumnB, ColumnC, ColumnD)
AS
(
	SELECT a.ColumnA, a.ColumnB, c.ColumnC, c.colD as ColumnD
	FROM table1 a join table2 c on a.colID = b.colID
)

SELECT ColumnA, ColumnD, count(ColumnB), sum(ColumnC)
FROM MyCTETable
Group By ColumnA, ColumnD
Order by ColumnA

-- using CTE to perform recursion
-- the example below is more intuitive than self join

WITH OrgReportCTE (ManagerID, EmployeeID, EmployeeName, MgrName, Level)
AS
(
	--anchor query
	SELECT e.ManagerID, e.EmployeeID, e.EmployeeName, NULL, 0
	FROM tblEmployee e
	where ManagerID is NULL
	
	UNION ALL
	-- recursive query
	SELECT e.managerID, e.EmployeeID, e.EmployeeName, o.EmployeeName, Level+1
	FROM tblEmployee e join OrgReportCTE o 
	ON e.managerID = o.employeeID
)
SELECT * from OrgReportCTE
OPTION (MAXRECURSION 3)

------------------------------------------------------------------------------				
--																			--
--																			--
--								Advanced Aggregates							--
--																			--
--																			--
------------------------------------------------------------------------------

-- GROUPING SETS

SELECT ParentProdCategory, ProdCategory,  Sum(ColumnD)
from table1
group by 
GROUPING SET (ParentProdCategory, ProdCategory, ()) -- Grouping set gives you aggregates at the set level you specify. The empty () returns cumulative sum of columnD

SELECT ParentProdCategory, ProdCategory,  Sum(ColumnD)
from table1
group by 
ROLLUP (ParentProdCategory, ProdCategory) -- ROLLUP returns aggregate at Parent category, and a break up of parent into child. It does NOT return aggregate at child category

SELECT ParentProdCategory, ProdCategory,  Sum(ColumnD)
from table1
group by 
CUBE (ParentProdCategory, ProdCategory) --CUBE returns every possible combination aggregates


------------------------------------------------------------------------------				
--																			--
--																			--
--								Programming 					            --
--																			--
--																			--
------------------------------------------------------------------------------

-- USE OF VARIABLES

-- here default variable is 'Toronto'
DECLARE @City Varchar(20) = 'Toronto' 

-- you can also declare a variable using SET
DECLARE @City Varchar(20) = 'Toronto'
SET @City = 'Austin' -- explicitly set the City variable

Select -- standard query language
from -- some table with joins if needed
where City = @City

-- variables can also be used as an output using SELECT
DECLARE @TotalSales money -- money is the variable type
SELECT @TotalSales = SUM(SalesRevenue)
FROM schemaname.SalesDetailTable

-- SIMPLE IF/ELSE

UPDATE schemaname.SalesDetailTable
SET OrderCancelledDate = GETDATE()
WHERE OrderID = 997

IF @@ROWCOUNT <1 OR @@ERROR <>0
BEGIN
	Print 'Order not found'
END
ELSE
BEGIN
	Print 'Order cancelled'
END

-- the other way of executing the above (not tested)

IF EXISTS (Select OrderID from schemaname.SalesDetailTable where OrderID = 997)
BEGIN
	UPDATE schemaname.SalesDetailTable
	SET OrderCancelledDate = GETDATE()
	WHERE OrderID = 997
END
ELSE
BEGIN
	Print 'Order cancelled'
END

-- STORED PROCEDURES

USE [database]
GO

CREATE PROC [spMyStoredProc] --or ALTER (if SP exists) 
	(
	@MinOrder INT = 0,
	@MaxOrder INT = NULL,
	@SalesPerson as nvarchar(max)
	)
AS
BEGIN

	SELECT SalesPerson, CustomerName, SalesRevenue
	from schemaname.tblSales
	WHERE 
	(@MinOrder IS NULL OR SalesRevenue >= @MinOrder) AND
	(@MaxOrder IS NULL OR SalesRevenue <= @MaxOrder) AND
	SalesPerson LIKE '%'+@SalesPerson+'%'
END

EXEC spMyStoredProc @MinOrder = 10, @SalesPerson = 'John Doe'	
