--DWD 层

package com.masu.sql

import org.apache.spark.sql.SparkSession --添加SPark依赖，避免报错

object dwd_electric_car_info {
  def main(args: Array[String]): Unit = {
    // 创建SparkSQL环境
    val ss: SparkSession = SparkSession.builder()
      .master("local[*]") // 本地多线程运行，生产环境删除
      .appName("DWD层-电动汽车销售数据清洗")
      .config("spark.sql.shuffle.partitions", "4") // 调整shuffle分区，生产环境按需设置
      .enableHiveSupport() // 开启Hive支持
      .getOrCreate()

    // 1. 创建DWD层数据库
    ss.sql("CREATE DATABASE IF NOT EXISTS dwd_electric_car")

    // 2. 建DWD明细表（清洗后，字段类型标准化）
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dwd_electric_car.dwd_electric_car_info (
        |    id BIGINT COMMENT '唯一标识ID',
        |    brand STRING COMMENT '电动汽车品牌',
        |    origin STRING COMMENT '电动汽车产地',
        |    production_volume BIGINT COMMENT '电动汽车产量',
        |    sales_volume BIGINT COMMENT '电动汽车销量',
        |    operating_kilometers DECIMAL(15,0) COMMENT '电动汽车运行公里数',
        |    production_date STRING COMMENT '电动汽车生产日期(YYYY-MM)',
        |    member_level STRING COMMENT '拥有者会员级别',
        |    original_price DECIMAL(15,0) COMMENT '电动汽车原价',
        |    sale_price DECIMAL(15,0) COMMENT '电动汽车售价',
        |    profit DECIMAL(15,0) COMMENT '电动汽车利润',
        |    transfer_status STRING COMMENT '电动汽车过户状态',
        |    manufacturer STRING COMMENT '电动汽车厂商',
        |    battery_capacity_kwh DECIMAL(10,2) COMMENT '电池容量(kWh)',
        |    range_nedc_cltc DECIMAL(10,2) COMMENT '续航里程(NEDC/CLTC)',
        |    charging_efficiency STRING COMMENT '充电效率',
        |    drive_type STRING COMMENT '驱动形式',
        |    market_share DECIMAL(20,7) COMMENT '市场份额',
        |    user_age BIGINT COMMENT '用户年龄',
        |    user_gender STRING COMMENT '用户性别',
        |    user_occupation STRING COMMENT '用户职业',
        |    user_habits STRING COMMENT '用户使用习惯',
        |    intelligence_level STRING COMMENT '电动汽车智能化水平'
        |)
        |COMMENT 'DWD层-电动汽车销售清洗后明细表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE
        |TBLPROPERTIES (
        |    'serialization.null.format'='',
        |    'mapreduce.output.fileoutputformat.compress'='true',
        |    'mapreduce.output.fileoutputformat.compress.codec'='org.apache.hadoop.io.compress.GzipCodec'
        |);
        |""".stripMargin)

    // 3. 插入数据：从ODS层清洗后插入，做类型转换+脏数据过滤
    ss.sql(
      """
        |INSERT OVERWRITE TABLE dwd_electric_car.dwd_electric_car_info
        |SELECT
        |    CAST(id AS BIGINT) AS id,
        |    brand,
        |    origin,
        |    CAST(production_volume AS BIGINT) AS production_volume,
        |    CAST(sales_volume AS BIGINT) AS sales_volume,
        |    CAST(operating_kilometers AS DECIMAL(15,0)) AS operating_kilometers,
        |    production_date,
        |    member_level,
        |    CAST(original_price AS DECIMAL(15,0)) AS original_price,
        |    CAST(sale_price AS DECIMAL(15,0)) AS sale_price,
        |    CAST(profit AS DECIMAL(15,0)) AS profit,
        |    transfer_status,
        |    manufacturer,
        |    CAST(battery_capacity_kwh AS DECIMAL(10,2)) AS battery_capacity_kwh,
        |    CAST(range_nedc_cltc AS DECIMAL(10,2)) AS range_nedc_cltc,
        |    charging_efficiency,
        |    drive_type,
        |    CAST(market_share AS DECIMAL(20,7)) AS market_share,
        |    CAST(user_age AS BIGINT) AS user_age,
        |    user_gender,
        |    user_occupation,
        |    user_habits,
        |    intelligence_level
        |FROM electric_car_sale.electric_car_sales
        |-- 脏数据过滤：非空日期、销量/产量≥0、价格≥0、年龄18-100岁
        |WHERE production_date IS NOT NULL
        |  AND production_date RLIKE '^\\d{4}-\\d{2}$'
        |  AND CAST(sales_volume AS BIGINT) >= 0
        |  AND CAST(production_volume AS BIGINT) >= 0
        |  AND CAST(original_price AS DECIMAL(15,0)) >= 0
        |  AND CAST(sale_price AS DECIMAL(15,0)) >= 0
        |  AND CAST(user_age AS BIGINT) BETWEEN 18 AND 100;
        |""".stripMargin)

    // 测试查询（生产环境注释）
    // ss.sql("SELECT * FROM dwd_electric_car.dwd_electric_car_info LIMIT 10").show()

    // 关闭资源
    ss.stop()
  }

}



--DWS 层

package com.masu.sql

import org.apache.spark.sql.SparkSession

object dws_electric_car_month_agg {
  def main(args: Array[String]): Unit = {
    // 创建SparkSQL环境
    val ss: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("DWS层-电动汽车销售月度基础聚合")
      .config("spark.sql.shuffle.partitions", "4")
      .enableHiveSupport()
      .getOrCreate()

    // 1. 创建DWS层数据库
    ss.sql("CREATE DATABASE IF NOT EXISTS dws_electric_car")

    // 2. 建DWS月度聚合表（核心维度：stat_month+origin+brand+产品特征）
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dws_electric_car.dws_electric_car_month_agg (
        |    stat_month STRING COMMENT '统计月份(YYYY-MM)',
        |    origin STRING COMMENT '产地',
        |    brand STRING COMMENT '电动汽车品牌',
        |    drive_type STRING COMMENT '驱动形式',
        |    charging_efficiency STRING COMMENT '充电效率',
        |    intelligence_level STRING COMMENT '智能化水平',
        |    transfer_status STRING COMMENT '过户状态',
        |    battery_capacity_kwh DECIMAL(10,2) COMMENT '电池容量(kWh)',
        |    range_nedc_cltc DECIMAL(10,2) COMMENT '续航里程(NEDC/CLTC)',
        |    user_age BIGINT COMMENT '用户年龄',
        |    user_gender STRING COMMENT '用户性别',
        |    user_occupation STRING COMMENT '用户职业',
        |    total_sales_volume BIGINT COMMENT '月度销量',
        |    total_production_volume BIGINT COMMENT '月度产量',
        |    total_original_price DECIMAL(20,2) COMMENT '月度原价总和',
        |    total_sale_price DECIMAL(20,2) COMMENT '月度售价总和',
        |    total_profit DECIMAL(20,2) COMMENT '月度利润总和',
        |    total_market_share DECIMAL(20,7) COMMENT '月度市场份额总和',
        |    customer_count BIGINT COMMENT '客户数量',
        |    avg_battery_capacity DECIMAL(10,2) COMMENT '平均电池容量',
        |    avg_range_nedc DECIMAL(10,2) COMMENT '平均续航里程'
        |)
        |COMMENT 'DWS层-电动汽车销售月度基础聚合表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE
        |TBLPROPERTIES (
        |    'serialization.null.format'='',
        |    'mapreduce.output.fileoutputformat.compress'='true',
        |    'mapreduce.output.fileoutputformat.compress.codec'='org.apache.hadoop.io.compress.GzipCodec'
        |);
        |""".stripMargin)

    // 3. 插入数据：从DWD层按维度聚合，计算基础指标
    ss.sql(
      """
        |INSERT OVERWRITE TABLE dws_electric_car.dws_electric_car_month_agg
        |SELECT
        |    production_date AS stat_month, -- 生产日期作为统计月份
        |    origin,
        |    brand,
        |    drive_type,
        |    charging_efficiency,
        |    intelligence_level,
        |    transfer_status,
        |    battery_capacity_kwh,
        |    range_nedc_cltc,
        |    user_age,
        |    user_gender,
        |    user_occupation,
        |    SUM(sales_volume) AS total_sales_volume,
        |    SUM(production_volume) AS total_production_volume,
        |    SUM(original_price) AS total_original_price,
        |    SUM(sale_price) AS total_sale_price,
        |    SUM(profit) AS total_profit,
        |    SUM(market_share) AS total_market_share,
        |    COUNT(DISTINCT id) AS customer_count, -- 按ID去重统计客户数
        |    AVG(battery_capacity_kwh) AS avg_battery_capacity,
        |    AVG(range_nedc_cltc) AS avg_range_nedc
        |FROM dwd_electric_car.dwd_electric_car_info
        |GROUP BY
        |    production_date, origin, brand, drive_type, charging_efficiency,
        |    intelligence_level, transfer_status, battery_capacity_kwh,
        |    range_nedc_cltc, user_age, user_gender, user_occupation;
        |""".stripMargin)

    // 测试查询（生产环境注释）
    // ss.sql("SELECT stat_month, origin, total_sales_volume FROM dws_electric_car.dws_electric_car_month_agg LIMIT 10").show()

    // 关闭资源
    ss.stop()
  }

}




--DM 层

package com.masu.sql

import org.apache.spark.sql.SparkSession

object dm_electric_car_analysis {
  def main(args: Array[String]): Unit = {
    // 创建SparkSQL环境
    val ss: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("DM层-电动汽车销售15张指标表生成")
      .config("spark.sql.shuffle.partitions", "8")
      .enableHiveSupport()
      .getOrCreate()

    // 1. 创建DM层数据库
    ss.sql("CREATE DATABASE IF NOT EXISTS dm_electric_car")
    //删除已经存在的表
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_monthly_sales_trend")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_quarterly_sales_trend")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_yearly_sales_trend")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_region_sales_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_customer_age_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_customer_gender_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_customer_occupation_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_price_range_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_profit_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_battery_capacity_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_range_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_drive_type_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_charging_efficiency_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_intelligence_level_analysis")
    ss.sql("DROP TABLE IF EXISTS dm_electric_car.dm_transfer_status_analysis")

    /****************************************************************************
     * 指标表1：月度销售趋势表（dm_monthly_sales_trend）- 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_monthly_sales_trend (
        |    stat_month STRING COMMENT '统计月份(YYYY-MM)',
        |    brand STRING COMMENT '电动汽车品牌',
        |    total_sales_volume BIGINT COMMENT '月度总销量',
        |    total_production_volume BIGINT COMMENT '月度总产量',
        |    avg_sale_price DECIMAL(10,2) COMMENT '平均售价',
        |    total_profit DECIMAL(15,2) COMMENT '月度总利润'
        |)
        |COMMENT '月度销售趋势分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_monthly_sales_trend
        |WITH monthly_base AS (
        |    SELECT
        |        stat_month,
        |        brand,
        |        SUM(total_sales_volume) AS total_sales_volume,
        |        SUM(total_production_volume) AS total_production_volume,
        |        ROUND(CASE WHEN IFNULL(SUM(customer_count),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(customer_count) END, 2) AS avg_sale_price,
        |        SUM(total_profit) AS total_profit
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |    GROUP BY stat_month, brand
        |)
        |SELECT
        |    stat_month,
        |    brand,
        |    total_sales_volume,
        |    total_production_volume,
        |    avg_sale_price,
        |    total_profit
        |FROM monthly_base;
        |""".stripMargin)

    /****************************************************************************
     * 指标表2：季度销售趋势表（dm_quarterly_sales_trend）- 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_quarterly_sales_trend (
        |    stat_quarter STRING COMMENT '统计季度(YYYY-Q)',
        |    brand STRING COMMENT '电动汽车品牌',
        |    total_sales_volume BIGINT COMMENT '季度总销量',
        |    total_production_volume BIGINT COMMENT '季度总产量',
        |    avg_sale_price DECIMAL(10,2) COMMENT '平均售价',
        |    total_profit DECIMAL(15,2) COMMENT '季度总利润'
        |)
        |COMMENT '季度销售趋势分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_quarterly_sales_trend
        |WITH quarter_base AS (
        |    SELECT
        |        CONCAT(SUBSTR(stat_month, 1, 4), '-Q', CEIL(SUBSTR(stat_month, 6, 2)/3)) AS stat_quarter,
        |        brand,
        |        SUM(total_sales_volume) AS total_sales_volume,
        |        SUM(total_production_volume) AS total_production_volume,
        |        ROUND(CASE WHEN IFNULL(SUM(customer_count),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(customer_count) END, 2) AS avg_sale_price,
        |        SUM(total_profit) AS total_profit
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |    GROUP BY CONCAT(SUBSTR(stat_month, 1, 4), '-Q', CEIL(SUBSTR(stat_month, 6, 2)/3)), brand
        |)
        |SELECT
        |    stat_quarter,
        |    brand,
        |    total_sales_volume,
        |    total_production_volume,
        |    avg_sale_price,
        |    total_profit
        |FROM quarter_base;
        |""".stripMargin)

    /****************************************************************************
     * 指标表3：年度销售趋势表（dm_yearly_sales_trend）- 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_yearly_sales_trend (
        |    stat_year STRING COMMENT '统计年份(YYYY)',
        |    brand STRING COMMENT '电动汽车品牌',
        |    total_sales_volume BIGINT COMMENT '年度总销量',
        |    total_production_volume BIGINT COMMENT '年度总产量',
        |    avg_sale_price DECIMAL(10,2) COMMENT '平均售价',
        |    total_profit DECIMAL(15,2) COMMENT '年度总利润'
        |)
        |COMMENT '年度销售趋势分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_yearly_sales_trend
        |WITH year_base AS (
        |    SELECT
        |        SUBSTR(stat_month, 1, 4) AS stat_year,
        |        brand,
        |        SUM(total_sales_volume) AS total_sales_volume,
        |        SUM(total_production_volume) AS total_production_volume,
        |        ROUND(CASE WHEN IFNULL(SUM(customer_count),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(customer_count) END, 2) AS avg_sale_price,
        |        SUM(total_profit) AS total_profit
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |    GROUP BY SUBSTR(stat_month, 1, 4), brand
        |)
        |SELECT
        |    stat_year,
        |    brand,
        |    total_sales_volume,
        |    total_production_volume,
        |    avg_sale_price,
        |    total_profit
        |FROM year_base;
        |""".stripMargin)

    /****************************************************************************
     * 指标表4：地区销售分析表（dm_region_sales_analysis）- 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_region_sales_analysis (
        |    origin STRING COMMENT '产地',
        |    brand STRING COMMENT '电动汽车品牌',
        |    stat_month STRING COMMENT '统计月份',
        |    total_sales_volume BIGINT COMMENT '地区月度销量',
        |    total_production_volume BIGINT COMMENT '地区月度产量',
        |    avg_sale_price DECIMAL(10,2) COMMENT '地区平均售价',
        |    market_share DECIMAL(12,8) COMMENT '地区市场份额',
        |    sales_rank INT COMMENT '地区销量排名'
        |)
        |COMMENT '地区销售分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_region_sales_analysis
        |WITH region_base AS (
        |    SELECT
        |        IFNULL(origin, '未知产地') AS origin,
        |        brand,
        |        stat_month,
        |        SUM(total_sales_volume) AS total_sales_volume,
        |        SUM(total_production_volume) AS total_production_volume,
        |        ROUND(CASE WHEN IFNULL(SUM(customer_count),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(customer_count) END, 2) AS avg_sale_price,
        |        ROUND(CASE WHEN IFNULL(COUNT(DISTINCT brand),0) = 0 THEN 0 ELSE SUM(total_market_share) / COUNT(DISTINCT brand) END, 8) AS market_share
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |    GROUP BY origin, brand, stat_month
        |)
        |SELECT
        |    origin,
        |    brand,
        |    stat_month,
        |    total_sales_volume,
        |    total_production_volume,
        |    avg_sale_price,
        |    market_share,
        |    RANK() OVER (PARTITION BY stat_month, origin ORDER BY total_sales_volume DESC) AS sales_rank
        |FROM region_base;
        |""".stripMargin)

    /****************************************************************************
     * 指标表5：客户年龄分析表（dm_customer_age_analysis）
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_customer_age_analysis (
        |    age_group STRING COMMENT '年龄段',
        |    stat_month STRING COMMENT '统计月份',
        |    customer_count BIGINT COMMENT '客户数量',
        |    total_sales_volume BIGINT COMMENT '销量',
        |    avg_purchase_price DECIMAL(10,2) COMMENT '平均购买价格',
        |    preferred_brand STRING COMMENT '偏好品牌',
        |    preferred_drive_type STRING COMMENT '偏好驱动类型'
        |)
        |COMMENT '客户年龄分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_customer_age_analysis
        |WITH age_group_base AS (
        |    SELECT
        |        CASE WHEN IFNULL(user_age,0) BETWEEN 18 AND 30 THEN '18-30岁'
        |             WHEN IFNULL(user_age,0) BETWEEN 31 AND 45 THEN '31-45岁'
        |             WHEN IFNULL(user_age,0) BETWEEN 46 AND 60 THEN '46-60岁'
        |             WHEN IFNULL(user_age,0) > 60 THEN '60岁以上'
        |             ELSE '未知年龄' END AS age_group,
        |        stat_month,
        |        IFNULL(brand, '未知品牌') AS brand,
        |        IFNULL(drive_type, '未知驱动') AS drive_type,
        |        total_sales_volume,
        |        customer_count,
        |        total_sale_price
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |),
        |age_agg AS (
        |    SELECT
        |        age_group,
        |        stat_month,
        |        SUM(customer_count) AS customer_count,
        |        SUM(total_sales_volume) AS total_sales_volume,
        |        ROUND(CASE WHEN IFNULL(SUM(customer_count),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(customer_count) END, 2) AS avg_purchase_price
        |    FROM age_group_base
        |    GROUP BY age_group, stat_month
        |),
        |age_brand AS (
        |    SELECT
        |        age_group,
        |        stat_month,
        |        brand,
        |        SUM(total_sales_volume) AS sales,
        |        RANK() OVER (PARTITION BY age_group, stat_month ORDER BY SUM(total_sales_volume) DESC) AS rk
        |    FROM age_group_base
        |    GROUP BY age_group, stat_month, brand
        |),
        |age_drive AS (
        |    SELECT
        |        age_group,
        |        stat_month,
        |        drive_type,
        |        SUM(total_sales_volume) AS sales,
        |        RANK() OVER (PARTITION BY age_group, stat_month ORDER BY SUM(total_sales_volume) DESC) AS rk
        |    FROM age_group_base
        |    GROUP BY age_group, stat_month, drive_type
        |)
        |SELECT
        |    a.age_group,
        |    a.stat_month,
        |    a.customer_count,
        |    a.total_sales_volume,
        |    a.avg_purchase_price,
        |    IFNULL(b.brand, '无偏好品牌') AS preferred_brand,
        |    IFNULL(d.drive_type, '无偏好驱动') AS preferred_drive_type
        |FROM age_agg a
        |LEFT JOIN age_brand b ON a.age_group = b.age_group AND a.stat_month = b.stat_month AND b.rk = 1
        |LEFT JOIN age_drive d ON a.age_group = d.age_group AND a.stat_month = d.stat_month AND d.rk = 1;
        |""".stripMargin)

    /****************************************************************************
     * 指标表6：客户性别分析表（dm_customer_gender_analysis）
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_customer_gender_analysis (
        |    user_gender STRING COMMENT '用户性别',
        |    stat_month STRING COMMENT '统计月份',
        |    customer_count BIGINT COMMENT '客户数量',
        |    total_sales_volume BIGINT COMMENT '销量',
        |    avg_purchase_price DECIMAL(10,2) COMMENT '平均购买价格',
        |    preferred_brand STRING COMMENT '偏好品牌'
        |)
        |COMMENT '客户性别分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_customer_gender_analysis
        |WITH gender_base AS (
        |    SELECT
        |        CASE WHEN user_gender IS NULL OR user_gender = '' THEN '未知性别' ELSE user_gender END AS user_gender,
        |        stat_month,
        |        IFNULL(brand, '未知品牌') AS brand,
        |        total_sales_volume,
        |        customer_count,
        |        total_sale_price
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |),
        |gender_agg AS (
        |    SELECT
        |        user_gender,
        |        stat_month,
        |        SUM(customer_count) AS customer_count,
        |        SUM(total_sales_volume) AS total_sales_volume,
        |        ROUND(CASE WHEN IFNULL(SUM(customer_count),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(customer_count) END, 2) AS avg_purchase_price
        |    FROM gender_base
        |    GROUP BY user_gender, stat_month
        |),
        |gender_brand AS (
        |    SELECT
        |        user_gender,
        |        stat_month,
        |        brand,
        |        SUM(total_sales_volume) AS sales,
        |        RANK() OVER (PARTITION BY user_gender, stat_month ORDER BY SUM(total_sales_volume) DESC) AS rk
        |    FROM gender_base
        |    GROUP BY user_gender, stat_month, brand
        |)
        |SELECT
        |    g.user_gender,
        |    g.stat_month,
        |    g.customer_count,
        |    g.total_sales_volume,
        |    g.avg_purchase_price,
        |    IFNULL(b.brand, '无偏好品牌') AS preferred_brand
        |FROM gender_agg g
        |LEFT JOIN gender_brand b ON g.user_gender = b.user_gender AND g.stat_month = b.stat_month AND b.rk = 1;
        |""".stripMargin)

    /****************************************************************************
     * 指标表7：客户职业分析表（dm_customer_occupation_analysis）
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_customer_occupation_analysis (
        |    user_occupation STRING COMMENT '用户职业',
        |    stat_month STRING COMMENT '统计月份',
        |    customer_count BIGINT COMMENT '客户数量',
        |    total_sales_volume BIGINT COMMENT '销量',
        |    avg_purchase_price DECIMAL(10,2) COMMENT '平均购买价格',
        |    preferred_brand STRING COMMENT '偏好品牌',
        |    purchase_ability_score DECIMAL(3,1) COMMENT '购买能力评分(1-10分)'
        |)
        |COMMENT '客户职业分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_customer_occupation_analysis
        |WITH occ_base AS (
        |    SELECT
        |        CASE WHEN user_occupation IS NULL OR user_occupation = '' THEN '未知职业' ELSE user_occupation END AS user_occupation,
        |        stat_month,
        |        IFNULL(brand, '未知品牌') AS brand,
        |        total_sales_volume,
        |        customer_count,
        |        total_sale_price,
        |        ROUND(CASE WHEN IFNULL(SUM(customer_count) OVER (PARTITION BY stat_month),0) = 0 THEN 0
        |              ELSE SUM(total_sale_price) OVER (PARTITION BY stat_month) / SUM(customer_count) OVER (PARTITION BY stat_month) END, 2) AS month_avg_price
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |),
        |occ_agg AS (
        |    SELECT
        |        user_occupation,
        |        stat_month,
        |        SUM(customer_count) AS customer_count,
        |        SUM(total_sales_volume) AS total_sales_volume,
        |        ROUND(CASE WHEN IFNULL(SUM(customer_count),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(customer_count) END, 2) AS avg_purchase_price,
        |        MAX(month_avg_price) AS month_avg_price
        |    FROM occ_base
        |    GROUP BY user_occupation, stat_month
        |),
        |occ_brand AS (
        |    SELECT
        |        user_occupation,
        |        stat_month,
        |        brand,
        |        SUM(total_sales_volume) AS sales,
        |        RANK() OVER (PARTITION BY user_occupation, stat_month ORDER BY SUM(total_sales_volume) DESC) AS rk
        |    FROM occ_base
        |    GROUP BY user_occupation, stat_month, brand
        |),
        |occ_score AS (
        |    SELECT
        |        *,
        |        ROUND(LEAST(GREATEST(CASE WHEN IFNULL(month_avg_price,0) = 0 THEN 5.0 ELSE (avg_purchase_price / month_avg_price) * 5 END, 1.0), 10.0), 1) AS purchase_ability_score
        |    FROM occ_agg
        |)
        |SELECT
        |    s.user_occupation,
        |    s.stat_month,
        |    s.customer_count,
        |    s.total_sales_volume,
        |    s.avg_purchase_price,
        |    IFNULL(b.brand, '无偏好品牌') AS preferred_brand,
        |    s.purchase_ability_score
        |FROM occ_score s
        |LEFT JOIN occ_brand b ON s.user_occupation = b.user_occupation AND s.stat_month = b.stat_month AND b.rk = 1;
        |""".stripMargin)

    /****************************************************************************
     * 指标表8：价格区间分析表（dm_price_range_analysis）- 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_price_range_analysis (
        |    brand STRING COMMENT '电动汽车品牌',
        |    price_range STRING COMMENT '价格区间',
        |    stat_month STRING COMMENT '统计月份',
        |    sales_volume BIGINT COMMENT '销量',
        |    avg_profit_margin DECIMAL(5,2) COMMENT '平均利润率',
        |    total_profit DECIMAL(15,2) COMMENT '总利润',
        |    customer_satisfaction_score DECIMAL(3,1) COMMENT '客户满意度评分(1-10分)'
        |)
        |COMMENT '价格区间分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_price_range_analysis
        |WITH price_base AS (
        |    SELECT
        |        IFNULL(brand, '未知品牌') AS brand,
        |        stat_month,
        |        total_sales_volume,
        |        total_sale_price,
        |        total_profit,
        |        CASE WHEN IFNULL(customer_count,0) = 0 THEN '未知价格区间'
        |             WHEN total_sale_price / customer_count < 200000 THEN '0-20万'
        |             WHEN total_sale_price / customer_count < 400000 THEN '20-40万'
        |             WHEN total_sale_price / customer_count < 600000 THEN '40-60万'
        |             ELSE '60万以上' END AS price_range,
        |        SUM(total_sales_volume) OVER (PARTITION BY stat_month) AS month_total_sales,  -- 全月总销量
        |        MAX(CASE WHEN total_sale_price > 0 THEN total_profit / total_sale_price * 100 ELSE 0 END)
        |            OVER (PARTITION BY stat_month) AS month_max_profit_margin,  -- 全月最高利润率（用于归一化）
        |        MIN(CASE WHEN total_sale_price > 0 THEN total_profit / total_sale_price * 100 ELSE 0 END)
        |            OVER (PARTITION BY stat_month) AS month_min_profit_margin   -- 全月最低利润率（用于归一化）
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |),
        |price_agg AS (
        |    SELECT
        |        brand,
        |        price_range,
        |        stat_month,
        |        SUM(total_sales_volume) AS sales_volume,  -- 该价格区间销量
        |        MAX(month_total_sales) AS month_total_sales,
        |        MAX(month_max_profit_margin) AS month_max_profit_margin,
        |        MAX(month_min_profit_margin) AS month_min_profit_margin,
        |        SUM(total_profit) AS total_profit,
        |        SUM(total_sale_price) AS total_sale_price,
        |        -- 计算该价格区间的平均利润率
        |        ROUND(CASE WHEN IFNULL(SUM(total_sale_price),0) = 0 THEN 0
        |                    ELSE SUM(total_profit) / SUM(total_sale_price) * 100 END, 2) AS avg_profit_margin
        |    FROM price_base
        |    GROUP BY brand, price_range, stat_month
        |),
        |price_calc AS (
        |    SELECT
        |        *,
        |        -- 1. 销量占比归一化（0-1）：该区间销量 / 全月总销量，无销量则为 0
        |        CASE WHEN month_total_sales = 0 THEN 0
        |             ELSE sales_volume / CAST(month_total_sales AS DECIMAL) END AS sales_ratio,
        |        -- 2. 利润率归一化（0-1）：(当前利润率 - 全月最低利润率) / (全月最高利润率 - 全月最低利润率)，避免负分
        |        CASE WHEN month_max_profit_margin = month_min_profit_margin THEN 0.5  -- 全月利润率一致，中间值
        |             ELSE (avg_profit_margin - month_min_profit_margin) / (month_max_profit_margin - month_min_profit_margin) END AS profit_ratio
        |    FROM price_agg
        |)
        |SELECT
        |    brand,
        |    price_range,
        |    stat_month,
        |    sales_volume,
        |    avg_profit_margin,
        |    total_profit,
        |    -- 优化后评分：销量占比60% + 利润率40%，归一化后加权，控制1.0-10.0分，保留1位小数
        |    ROUND(
        |        LEAST(
        |            GREATEST(
        |                (sales_ratio * 0.6 + profit_ratio * 0.4) * 10,  -- 归一化后加权，放大为0-10分
        |                1.0
        |            ),
        |            10.0
        |        ),
        |        1
        |    ) AS customer_satisfaction_score
        |FROM price_calc;
        |""".stripMargin)

    /****************************************************************************
     * 指标表9：利润分析表（dm_profit_analysis）- 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_profit_analysis (
        |    brand STRING COMMENT '电动汽车品牌',
        |    stat_month STRING COMMENT '统计月份',
        |    total_revenue DECIMAL(15,2) COMMENT '总收入(总售价)',
        |    total_cost DECIMAL(15,2) COMMENT '总成本(总收入-总利润)',
        |    total_profit DECIMAL(15,2) COMMENT '总利润',
        |    profit_margin DECIMAL(5,2) COMMENT '利润率(利润/收入*100)',
        |    profit_per_unit DECIMAL(10,2) COMMENT '单车利润(总利润/总销量)'
        |)
        |COMMENT '利润分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_profit_analysis
        |WITH profit_base AS (
        |    SELECT
        |        IFNULL(brand, '未知品牌') AS brand,
        |        stat_month,
        |        SUM(total_sale_price) AS total_revenue,
        |        SUM(total_profit) AS total_profit,
        |        SUM(total_sales_volume) AS total_sales,
        |        SUM(total_sale_price) - SUM(total_profit) AS total_cost,
        |        ROUND(CASE WHEN IFNULL(SUM(total_sale_price),0) = 0 THEN 0 ELSE SUM(total_profit) / SUM(total_sale_price) * 100 END, 2) AS profit_margin,
        |        ROUND(CASE WHEN IFNULL(SUM(total_sales_volume),0) = 0 THEN 0 ELSE SUM(total_profit) / SUM(total_sales_volume) END, 2) AS profit_per_unit
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |    GROUP BY brand, stat_month
        |)
        |SELECT
        |    brand,
        |    stat_month,
        |    total_revenue,
        |    total_cost,
        |    total_profit,
        |    profit_margin,
        |    profit_per_unit
        |FROM profit_base;
        |""".stripMargin)

    /****************************************************************************
     * 指标表10：电池容量分析表（dm_battery_capacity_analysis）- 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_battery_capacity_analysis (
        |    brand STRING COMMENT '电动汽车品牌',
        |    battery_range STRING COMMENT '电池容量区间(kWh)',
        |    stat_month STRING COMMENT '统计月份',
        |    sales_volume BIGINT COMMENT '销量',
        |    avg_range_nedc DECIMAL(10,1) COMMENT '平均续航里程',
        |    avg_sale_price DECIMAL(10,2) COMMENT '平均售价',
        |    customer_preference_score DECIMAL(3,1) COMMENT '客户偏好评分(1-10分)'
        |)
        |COMMENT '电池容量分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_battery_capacity_analysis
        |WITH battery_base AS (
        |    SELECT
        |        IFNULL(brand, '未知品牌') AS brand,
        |        stat_month,
        |        IFNULL(battery_capacity_kwh, 0) AS battery_capacity_kwh,
        |        IFNULL(range_nedc_cltc, 0) AS range_nedc_cltc,
        |        total_sales_volume,
        |        total_sale_price,
        |        -- 1. 计算当月所有电池容量的总销量（用于销量占比）
        |        SUM(total_sales_volume) OVER (PARTITION BY stat_month) AS month_total_sales,
        |        -- 2. 计算当月所有车型的平均续航（用于续航匹配度）
        |        AVG(range_nedc_cltc) OVER (PARTITION BY stat_month) AS month_avg_range,
        |        -- 3. 计算当月所有车型的平均售价（用于价格性价比）
        |        AVG(total_sale_price / customer_count) OVER (PARTITION BY stat_month) AS month_avg_price,
        |        -- 4. 电池容量区间划分（保留原逻辑）
        |        CASE WHEN IFNULL(battery_capacity_kwh, 0) < 50 AND IFNULL(battery_capacity_kwh, 0) > 0 THEN '0-50kWh'
        |             WHEN IFNULL(battery_capacity_kwh, 0) < 80 THEN '50-80kWh'
        |             WHEN IFNULL(battery_capacity_kwh, 0) < 110 THEN '80-110kWh'
        |             WHEN IFNULL(battery_capacity_kwh, 0) >= 110 THEN '110kWh以上'
        |             ELSE '未知容量区间' END AS battery_range
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |    -- 过滤无效数据：销量、续航、售价非空，避免计算错误
        |    WHERE total_sales_volume IS NOT NULL
        |      AND range_nedc_cltc IS NOT NULL
        |      AND total_sale_price IS NOT NULL
        |      AND customer_count > 0
        |),
        |battery_agg AS (
        |    SELECT
        |        brand,
        |        battery_range,
        |        stat_month,
        |        SUM(total_sales_volume) AS sales_volume,
        |        MAX(month_total_sales) AS month_total_sales,
        |        MAX(month_avg_range) AS month_avg_range,
        |        MAX(month_avg_price) AS month_avg_price,
        |        -- 1. 平均续航（保留1位小数）
        |        ROUND(AVG(range_nedc_cltc), 1) AS avg_range_nedc,
        |        -- 2. 平均售价（保留2位小数）
        |        ROUND(SUM(total_sale_price) / SUM(total_sales_volume), 2) AS avg_sale_price,
        |        -- 3. 销量占比（当前电池容量销量/当月总销量，保留4位小数，提升精度）
        |        ROUND(CASE WHEN IFNULL(MAX(month_total_sales), 0) = 0 THEN 0
        |              ELSE SUM(total_sales_volume) / CAST(MAX(month_total_sales) AS DECIMAL(18,4)) END, 4) AS sales_ratio
        |    FROM battery_base
        |    GROUP BY brand, battery_range, stat_month
        |),
        |battery_calc AS (
        |    SELECT
        |        *,
        |        -- 优化后：多维度加权客户偏好评分（1-10分，权重更合理）
        |        ROUND(
        |            LEAST(  -- 限制最高分不超过10
        |                GREATEST(  -- 限制最低分不低于1
        |                    -- 维度1：电池容量合理性（40%权重）- 容量越大、越贴合主流需求，得分越高
        |                    CASE WHEN battery_range = '110kWh以上' THEN 9.0
        |                         WHEN battery_range = '80-110kWh' THEN 8.0
        |                         WHEN battery_range = '50-80kWh' THEN 7.0
        |                         WHEN battery_range = '0-50kWh' THEN 5.0
        |                         ELSE 4.0 END * 0.4
        |                    +
        |                    -- 维度2：续航匹配度（30%权重）- 续航越接近当月均值，得分越高
        |                    CASE WHEN IFNULL(month_avg_range, 0) = 0 THEN 5.0
        |                         ELSE 10 - ABS(avg_range_nedc - month_avg_range) / month_avg_range * 10 END * 0.3
        |                    +
        |                    -- 维度3：价格性价比（20%权重）- 售价越接近当月均价，性价比越高，得分越高
        |                    CASE WHEN IFNULL(month_avg_price, 0) = 0 THEN 5.0
        |                         ELSE 10 - ABS(avg_sale_price - month_avg_price) / month_avg_price * 10 END * 0.2
        |                    +
        |                    -- 维度4：销量认可度（10%权重）- 销量占比越高，市场认可度越高
        |                    sales_ratio * 10 * 0.1,
        |                1.0),  -- 最低分1分（避免极端数据，但不会全为1.0）
        |            10.0),  -- 最高分10分
        |        1  -- 保留1位小数
        |        ) AS customer_preference_score
        |    FROM battery_agg
        |)
        |SELECT
        |    brand,
        |    battery_range,
        |    stat_month,
        |    sales_volume,
        |    avg_range_nedc,
        |    avg_sale_price,
        |    customer_preference_score
        |FROM battery_calc;
        |""".stripMargin)

    /****************************************************************************
     * 指标表11：续航里程分析表（dm_range_analysis）- 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_range_analysis (
        |    brand STRING COMMENT '电动汽车品牌',
        |    range_category STRING COMMENT '续航类别(km)',
        |    stat_month STRING COMMENT '统计月份',
        |    sales_volume BIGINT COMMENT '销量',
        |    avg_battery_capacity DECIMAL(10,1) COMMENT '平均电池容量',
        |    avg_sale_price DECIMAL(10,2) COMMENT '平均售价',
        |    customer_satisfaction DECIMAL(3,1) COMMENT '客户满意度(1-10分)',
        |    charging_frequency DECIMAL(3,1) COMMENT '充电频率(次/周)'
        |)
        |COMMENT '续航里程分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_range_analysis
        |WITH range_base AS (
        |    SELECT
        |        IFNULL(brand, '未知品牌') AS brand,
        |        stat_month,
        |        IFNULL(range_nedc_cltc,0) AS range_nedc_cltc,
        |        IFNULL(battery_capacity_kwh,0) AS battery_capacity_kwh,
        |        total_sales_volume,
        |        total_sale_price,
        |        SUM(total_sales_volume) OVER (PARTITION BY stat_month) AS month_total_sales,
        |        MAX(range_nedc_cltc) OVER (PARTITION BY stat_month) AS month_max_range,
        |        CASE WHEN IFNULL(range_nedc_cltc,0) < 300 AND IFNULL(range_nedc_cltc,0) > 0 THEN '0-300km'
        |             WHEN IFNULL(range_nedc_cltc,0) < 500 THEN '300-500km'
        |             WHEN IFNULL(range_nedc_cltc,0) < 700 THEN '500-700km'
        |             WHEN IFNULL(range_nedc_cltc,0) >= 700 THEN '700km以上'
        |             ELSE '未知续航区间' END AS range_category
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |),
        |range_agg AS (
        |    SELECT
        |        brand,
        |        range_category,
        |        stat_month,
        |        SUM(total_sales_volume) AS sales_volume,
        |        ROUND(AVG(battery_capacity_kwh), 1) AS avg_battery_capacity,
        |        ROUND(CASE WHEN IFNULL(SUM(total_sales_volume),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(total_sales_volume) END, 2) AS avg_sale_price,
        |        ROUND(AVG(range_nedc_cltc),1) AS avg_range,
        |        MAX(month_total_sales) AS month_total_sales,
        |        MAX(month_max_range) AS month_max_range
        |    FROM range_base
        |    GROUP BY brand, range_category, stat_month
        |),
        |range_calc AS (
        |    SELECT
        |        *,
        |        ROUND(LEAST(GREATEST(10 - (avg_range/100*8),1.0),10.0),1) AS charging_frequency,
        |        ROUND(LEAST(GREATEST(CASE WHEN IFNULL(month_max_range,0) = 0 THEN 5.0
        |              ELSE (avg_range/month_max_range)*7 + (sales_volume/CAST(month_total_sales AS DECIMAL))*3 END,1.0),10.0),1) AS customer_satisfaction
        |    FROM range_agg
        |)
        |SELECT
        |    brand,
        |    range_category,
        |    stat_month,
        |    sales_volume,
        |    avg_battery_capacity,
        |    avg_sale_price,
        |    customer_satisfaction,
        |    charging_frequency
        |FROM range_calc;
        |""".stripMargin)

    /****************************************************************************
     * 指标表12：驱动形式分析表（dm_drive_type_analysis）- 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_drive_type_analysis (
        |    brand STRING COMMENT '电动汽车品牌',
        |    drive_type STRING COMMENT '驱动形式',
        |    stat_month STRING COMMENT '统计月份',
        |    sales_volume BIGINT COMMENT '销量',
        |    avg_sale_price DECIMAL(10,2) COMMENT '平均售价',
        |    performance_score DECIMAL(3,1) COMMENT '性能评分(1-10分)',
        |    energy_efficiency DECIMAL(3,1) COMMENT '能源效率(1-10分)'
        |)
        |COMMENT '驱动形式分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_drive_type_analysis
        |WITH drive_base AS (
        |    SELECT
        |        IFNULL(brand, '未知品牌') AS brand,
        |        CASE WHEN drive_type IS NULL OR drive_type = '' THEN '未知驱动' ELSE drive_type END AS drive_type,
        |        stat_month,
        |        total_sales_volume,
        |        total_sale_price,
        |        SUM(total_sales_volume) OVER (PARTITION BY stat_month) AS month_total_sales
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |),
        |drive_agg AS (
        |    SELECT
        |        brand,
        |        drive_type,
        |        stat_month,
        |        SUM(total_sales_volume) AS sales_volume,
        |        MAX(month_total_sales) AS month_total_sales,
        |        ROUND(CASE WHEN IFNULL(SUM(total_sales_volume),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(total_sales_volume) END, 2) AS avg_sale_price,
        |        CASE WHEN drive_type = '四驱' THEN 9.0
        |             WHEN drive_type = '后驱' THEN 7.0
        |             WHEN drive_type = '前驱' THEN 5.0
        |             ELSE 4.0 END AS performance_score,
        |        CASE WHEN drive_type = '前驱' THEN 9.0
        |             WHEN drive_type = '后驱' THEN 7.0
        |             WHEN drive_type = '四驱' THEN 5.0
        |             ELSE 4.0 END AS energy_efficiency
        |    FROM drive_base
        |    GROUP BY brand, drive_type, stat_month
        |)
        |SELECT
        |    brand,
        |    drive_type,
        |    stat_month,
        |    sales_volume,
        |    avg_sale_price,
        |    performance_score,
        |    energy_efficiency
        |FROM drive_agg;
        |""".stripMargin)

    /****************************************************************************
     * 指标表13：充电效率分析表（dm_charging_efficiency_analysis）- 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_charging_efficiency_analysis (
        |    brand STRING COMMENT '电动汽车品牌',
        |    charging_efficiency_range STRING COMMENT '充电效率区间',
        |    stat_month STRING COMMENT '统计月份',
        |    sales_volume BIGINT COMMENT '销量',
        |    customer_satisfaction DECIMAL(3,1) COMMENT '客户满意度(1-10分)',
        |    avg_sale_price DECIMAL(10,2) COMMENT '平均售价',
        |    market_acceptance DECIMAL(3,1) COMMENT '市场接受度(1-10分)'
        |)
        |COMMENT '充电效率分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_charging_efficiency_analysis
        |WITH charge_base AS (
        |    SELECT
        |        IFNULL(brand, '未知品牌') AS brand,
        |        CASE WHEN charging_efficiency IS NULL OR charging_efficiency = '' THEN '未知充电类型'
        |             WHEN charging_efficiency = '快充' THEN '快充'
        |             WHEN charging_efficiency = '慢充' THEN '慢充'
        |             ELSE '混合充' END AS charging_efficiency_range,
        |        stat_month,
        |        total_sales_volume,
        |        total_sale_price,
        |        customer_count
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |    WHERE total_sales_volume > 0 AND total_sale_price > 0 AND customer_count > 0
        |),
        |charge_agg AS (
        |    SELECT
        |        brand,
        |        charging_efficiency_range,
        |        stat_month,
        |        SUM(total_sales_volume) AS sales_volume,
        |        ROUND(SUM(total_sale_price) / SUM(total_sales_volume), 2) AS avg_sale_price,
        |        -- 计算当月该充电类型的总销量（用于相对销量计算）
        |        SUM(SUM(total_sales_volume)) OVER (PARTITION BY stat_month, charging_efficiency_range) AS type_total_sales
        |    FROM charge_base
        |    GROUP BY brand, charging_efficiency_range, stat_month
        |),
        |charge_calc AS (
        |    SELECT
        |        *,
        |        -- 客户满意度 = 充电效率体验分 (60%) + 相对销量分 (40%)
        |        ROUND(
        |            LEAST(
        |                GREATEST(
        |                    -- 维度1：充电效率基础体验分
        |                    CASE WHEN charging_efficiency_range = '快充' THEN 9.0
        |                         WHEN charging_efficiency_range = '混合充' THEN 7.0
        |                         WHEN charging_efficiency_range = '慢充' THEN 5.0
        |                         ELSE 4.0 END * 0.6
        |                    +
        |                    -- 维度2：相对销量分（当前品牌销量 / 该类型总销量）
        |                    CASE WHEN type_total_sales = 0 THEN 0
        |                         ELSE (sales_volume / CAST(type_total_sales AS DECIMAL(18,4))) * 10 END * 0.4,
        |                1.0),
        |            10.0),
        |        1) AS customer_satisfaction,
        |        -- 市场接受度 = 该充电类型总销量的归一化得分
        |        ROUND(
        |            LEAST(
        |                GREATEST(
        |                    (type_total_sales / MAX(type_total_sales) OVER (PARTITION BY stat_month)) * 10,
        |                1.0),
        |            10.0),
        |        1) AS market_acceptance
        |    FROM charge_agg
        |)
        |SELECT
        |    brand,
        |    charging_efficiency_range,
        |    stat_month,
        |    sales_volume,
        |    customer_satisfaction,
        |    avg_sale_price,
        |    market_acceptance
        |FROM charge_calc;
        |""".stripMargin)

    /****************************************************************************
     * 指标表14：智能化水平分析表 - 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_intelligence_level_analysis (
        |    brand STRING COMMENT '电动汽车品牌',
        |    intelligence_level STRING COMMENT '智能化水平',
        |    stat_month STRING COMMENT '统计月份',
        |    sales_volume BIGINT COMMENT '销量',
        |    avg_sale_price DECIMAL(10,2) COMMENT '平均售价',
        |    future_potential DECIMAL(3,1) COMMENT '未来潜力评分(1-10分)'
        |)
        |COMMENT '智能化水平分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_intelligence_level_analysis
        |WITH intel_base AS (
        |    SELECT
        |        IFNULL(brand, '未知品牌') AS brand,
        |        CASE WHEN intelligence_level IS NULL OR intelligence_level = '' THEN '未知智能化水平' ELSE intelligence_level END AS intelligence_level,
        |        stat_month,
        |        total_sales_volume,
        |        total_sale_price,
        |        SUM(total_sales_volume) OVER (PARTITION BY stat_month) AS month_total_sales
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |),
        |intel_agg AS (
        |    SELECT
        |        brand,
        |        intelligence_level,
        |        stat_month,
        |        SUM(total_sales_volume) AS sales_volume,
        |        MAX(month_total_sales) AS month_total_sales,
        |        ROUND(CASE WHEN IFNULL(SUM(total_sales_volume),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(total_sales_volume) END, 2) AS avg_sale_price
        |    FROM intel_base
        |    GROUP BY brand, intelligence_level, stat_month
        |),
        |intel_calc AS (
        |    SELECT
        |        *,
        |        ROUND(LEAST(GREATEST(
        |            CASE WHEN intelligence_level LIKE '%L5%' THEN 8.0
        |                 WHEN intelligence_level LIKE '%L4%' THEN 6.0
        |                 WHEN intelligence_level LIKE '%L3%' THEN 4.0
        |                 WHEN intelligence_level LIKE '%L2%' THEN 2.0
        |                 ELSE 1.0 END + (CASE WHEN month_total_sales = 0 THEN 0 ELSE sales_volume/CAST(month_total_sales AS DECIMAL) END),
        |        1.0),10.0),1) AS future_potential
        |    FROM intel_agg
        |)
        |SELECT
        |    brand,
        |    intelligence_level,
        |    stat_month,
        |    sales_volume,
        |    avg_sale_price,
        |    future_potential
        |FROM intel_calc;
        |""".stripMargin)

    /****************************************************************************
     * 指标表15：过户状态分析表 - 新增品牌维度
     ***************************************************************************/
    ss.sql(
      """
        |CREATE TABLE IF NOT EXISTS dm_electric_car.dm_transfer_status_analysis (
        |    brand STRING COMMENT '电动汽车品牌',
        |    transfer_status STRING COMMENT '过户状态',
        |    stat_month STRING COMMENT '统计月份',
        |    transfer_count BIGINT COMMENT '过户数量',
        |    avg_transfer_price DECIMAL(10,2) COMMENT '平均过户价格',
        |    transfer_reason_analysis STRING COMMENT '过户原因分析'
        |)
        |COMMENT '过户状态分析表'
        |ROW FORMAT DELIMITED
        |FIELDS TERMINATED BY '\t'
        |STORED AS TEXTFILE;
        |""".stripMargin)

    ss.sql(
      """
        |INSERT OVERWRITE TABLE dm_electric_car.dm_transfer_status_analysis
        |WITH transfer_base AS (
        |    SELECT
        |        IFNULL(brand, '未知品牌') AS brand,
        |        CASE WHEN transfer_status IS NULL OR transfer_status = '' THEN '未知过户状态' ELSE transfer_status END AS transfer_status,
        |        stat_month,
        |        total_sales_volume,
        |        total_sale_price,
        |        SUM(total_sales_volume) OVER (PARTITION BY stat_month) AS month_total_sales
        |    FROM dws_electric_car.dws_electric_car_month_agg
        |),
        |transfer_agg AS (
        |    SELECT
        |        brand,
        |        transfer_status,
        |        stat_month,
        |        SUM(total_sales_volume) AS transfer_count,
        |        MAX(month_total_sales) AS month_total_sales,
        |        ROUND(CASE WHEN IFNULL(SUM(total_sales_volume),0) = 0 THEN 0 ELSE SUM(total_sale_price) / SUM(total_sales_volume) * 0.8 END, 2) AS avg_transfer_price
        |    FROM transfer_base
        |    GROUP BY brand, transfer_status, stat_month
        |),
        |transfer_calc AS (
        |    SELECT
        |        *,
        |        CASE WHEN transfer_status = '未过户' THEN '新车未过户，无转让需求，车辆保值率高'
        |             WHEN transfer_status = '已过户一次' THEN '首次过户，多为个人换车/资金周转，车辆使用周期较短'
        |             WHEN transfer_status = '已过户两次' THEN '二次过户，车辆使用频率较高/性价比偏低，保值率有所下降'
        |             WHEN transfer_status = '已过户三次' THEN '三次过户，车辆流通性强，保值率低，建议核查车况'
        |             WHEN transfer_status = '已过户三次及以上' THEN '多次过户，故障概率偏高，保值率极低，不建议入手'
        |             WHEN transfer_status = '未知过户状态' THEN '过户状态未登记，建议核查原始车辆档案'
        |             ELSE CONCAT(transfer_status, '，该过户次数下车辆流通性中等，保值率随过户次数增加降低') END AS transfer_reason_analysis
        |    FROM transfer_agg
        |)
        |SELECT
        |    brand,
        |    transfer_status,
        |    stat_month,
        |    transfer_count,
        |    avg_transfer_price,
        |    transfer_reason_analysis
        |FROM transfer_calc;
        |""".stripMargin)

    // 测试查询验证结果（生产环境可注释）
    ss.sql("SELECT * FROM dm_electric_car.dm_monthly_sales_trend LIMIT 5").show()
    ss.sql("SELECT * FROM dm_electric_car.dm_battery_capacity_analysis LIMIT 5").show()
    ss.sql("SELECT * FROM dm_electric_car.dm_intelligence_level_analysis LIMIT 5").show()
    ss.sql("SELECT * FROM dm_electric_car.dm_transfer_status_analysis LIMIT 5").show()

    // 关闭资源
    ss.stop()
  }
}