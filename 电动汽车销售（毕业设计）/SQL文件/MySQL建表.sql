DROP TABLE IF EXISTS `dm_monthly_sales_trend`;

DROP TABLE IF EXISTS `dm_quarterly_sales_trend`;

DROP TABLE IF EXISTS `dm_yearly_sales_trend`;

DROP TABLE IF EXISTS `dm_region_sales_analysis`;

DROP TABLE IF EXISTS `dm_customer_age_analysis`;

DROP TABLE IF EXISTS `dm_customer_gender_analysis`;

DROP TABLE IF EXISTS `dm_customer_occupation_analysis`;

DROP TABLE IF EXISTS `dm_price_range_analysis`;

DROP TABLE IF EXISTS `dm_profit_analysis`;

DROP TABLE IF EXISTS `dm_battery_capacity_analysis`;

DROP TABLE IF EXISTS `dm_range_analysis`;

DROP TABLE IF EXISTS `dm_drive_type_analysis`;

DROP TABLE IF EXISTS `dm_charging_efficiency_analysis`;

DROP TABLE IF EXISTS `dm_intelligence_level_analysis`;

DROP TABLE IF EXISTS `dm_transfer_status_analysis`;








-- 1. 月度销售趋势表
DROP TABLE IF EXISTS `dm_monthly_sales_trend`;

CREATE TABLE `dm_monthly_sales_trend` (
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份(YYYY-MM)',
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `total_sales_volume` BIGINT COMMENT '月度总销量',
  `total_production_volume` BIGINT COMMENT '月度总产量',
  `avg_sale_price` DECIMAL(10,2) COMMENT '平均售价',
  `total_profit` DECIMAL(15,2) COMMENT '月度总利润',
  PRIMARY KEY (`stat_month`, `brand`)  -- 主键调整为月份+品牌
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='月度销售趋势分析表';


-- 2. 季度销售趋势表
DROP TABLE IF EXISTS `dm_quarterly_sales_trend`;

CREATE TABLE `dm_quarterly_sales_trend` (
  `stat_quarter` VARCHAR(20) NOT NULL COMMENT '统计季度(YYYY-Q)',
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `total_sales_volume` BIGINT COMMENT '季度总销量',
  `total_production_volume` BIGINT COMMENT '季度总产量',
  `avg_sale_price` DECIMAL(10,2) COMMENT '平均售价',
  `total_profit` DECIMAL(15,2) COMMENT '季度总利润',
  PRIMARY KEY (`stat_quarter`, `brand`)  -- 主键调整为季度+品牌
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='季度销售趋势分析表';

-- 3. 年度销售趋势表
DROP TABLE IF EXISTS `dm_yearly_sales_trend`;

CREATE TABLE `dm_yearly_sales_trend` (
  `stat_year` VARCHAR(20) NOT NULL COMMENT '统计年份(YYYY)',
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `total_sales_volume` BIGINT COMMENT '年度总销量',
  `total_production_volume` BIGINT COMMENT '年度总产量',
  `avg_sale_price` DECIMAL(10,2) COMMENT '平均售价',
  `total_profit` DECIMAL(15,2) COMMENT '年度总利润',
  PRIMARY KEY (`stat_year`, `brand`)  -- 主键调整为年份+品牌
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='年度销售趋势分析表';

-- 4. 地区销售分析表
DROP TABLE IF EXISTS `dm_region_sales_analysis`;

CREATE TABLE `dm_region_sales_analysis` (
  `origin` VARCHAR(100) NOT NULL COMMENT '产地',
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `total_sales_volume` BIGINT COMMENT '地区月度销量',
  `total_production_volume` BIGINT COMMENT '地区月度产量',
  `avg_sale_price` DECIMAL(10,2) COMMENT '地区平均售价',
  `market_share` DECIMAL(12,8) COMMENT '地区市场份额',
  `sales_rank` INT COMMENT '地区销量排名',
  PRIMARY KEY (`origin`, `brand`, `stat_month`)  
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='地区销售分析表';

-- 5. 客户年龄分析表
DROP TABLE IF EXISTS `dm_customer_age_analysis`;

CREATE TABLE `dm_customer_age_analysis` (
  `age_group` VARCHAR(50) NOT NULL COMMENT '年龄段',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `customer_count` BIGINT COMMENT '客户数量',
  `total_sales_volume` BIGINT COMMENT '销量',
  `avg_purchase_price` DECIMAL(10,2) COMMENT '平均购买价格',
  `preferred_brand` VARCHAR(100) COMMENT '偏好品牌',
  `preferred_drive_type` VARCHAR(100) COMMENT '偏好驱动类型',
  PRIMARY KEY (`age_group`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户年龄分析表';

-- 6. 客户性别分析表
DROP TABLE IF EXISTS `dm_customer_gender_analysis`;

CREATE TABLE `dm_customer_gender_analysis` (
  `user_gender` VARCHAR(10) NOT NULL COMMENT '用户性别',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `customer_count` BIGINT COMMENT '客户数量',
  `total_sales_volume` BIGINT COMMENT '销量',
  `avg_purchase_price` DECIMAL(10,2) COMMENT '平均购买价格',
  `preferred_brand` VARCHAR(100) COMMENT '偏好品牌',
  PRIMARY KEY (`user_gender`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户性别分析表';

-- 7. 客户职业分析表
DROP TABLE IF EXISTS `dm_customer_occupation_analysis`;

CREATE TABLE `dm_customer_occupation_analysis` (
  `user_occupation` VARCHAR(100) NOT NULL COMMENT '用户职业',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `customer_count` BIGINT COMMENT '客户数量',
  `total_sales_volume` BIGINT COMMENT '销量',
  `avg_purchase_price` DECIMAL(10,2) COMMENT '平均购买价格',
  `preferred_brand` VARCHAR(100) COMMENT '偏好品牌',
  `purchase_ability_score` DECIMAL(3,1) COMMENT '购买能力评分(1-10分)',
  PRIMARY KEY (`user_occupation`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户职业分析表';

-- 8. 价格区间分析表
DROP TABLE IF EXISTS `dm_price_range_analysis`;

CREATE TABLE `dm_price_range_analysis` (
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `price_range` VARCHAR(50) NOT NULL COMMENT '价格区间',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `sales_volume` BIGINT COMMENT '销量',
  `avg_profit_margin` DECIMAL(5,2) COMMENT '平均利润率',
  `total_profit` DECIMAL(15,2) COMMENT '总利润',
  `customer_satisfaction_score` DECIMAL(3,1) COMMENT '客户满意度评分(1-10分)',
  PRIMARY KEY (`brand`, `price_range`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='价格区间分析表';

-- 9. 利润分析表
DROP TABLE IF EXISTS `dm_profit_analysis`;

CREATE TABLE `dm_profit_analysis` (
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `total_revenue` DECIMAL(15,2) COMMENT '总收入(总售价)',
  `total_cost` DECIMAL(15,2) COMMENT '总成本(总收入-总利润)',
  `total_profit` DECIMAL(15,2) COMMENT '总利润',
  `profit_margin` DECIMAL(5,2) COMMENT '利润率(利润/收入*100)',
  `profit_per_unit` DECIMAL(10,2) COMMENT '单车利润(总利润/总销量)',
  PRIMARY KEY (`brand`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='利润分析表';

-- 10. 电池容量分析表
DROP TABLE IF EXISTS `dm_battery_capacity_analysis`;

CREATE TABLE `dm_battery_capacity_analysis` (
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `battery_range` VARCHAR(50) NOT NULL COMMENT '电池容量区间(kWh)',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `sales_volume` BIGINT COMMENT '销量',
  `avg_range_nedc` DECIMAL(10,1) COMMENT '平均续航里程',
  `avg_sale_price` DECIMAL(10,2) COMMENT '平均售价',
  `customer_preference_score` DECIMAL(3,1) COMMENT '客户偏好评分(1-10分)',
  PRIMARY KEY (`brand`, `battery_range`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='电池容量分析表';


-- 11. 续航里程分析表
DROP TABLE IF EXISTS `dm_range_analysis`;

CREATE TABLE `dm_range_analysis` (
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `range_category` VARCHAR(50) NOT NULL COMMENT '续航类别(km)',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `sales_volume` BIGINT COMMENT '销量',
  `avg_battery_capacity` DECIMAL(10,1) COMMENT '平均电池容量',
  `avg_sale_price` DECIMAL(10,2) COMMENT '平均售价',
  `customer_satisfaction` DECIMAL(3,1) COMMENT '客户满意度(1-10分)',
  `charging_frequency` DECIMAL(3,1) COMMENT '充电频率(次/周)',
  PRIMARY KEY (`brand`, `range_category`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='续航里程分析表';

-- 12. 驱动形式分析表
DROP TABLE IF EXISTS `dm_drive_type_analysis`;

CREATE TABLE `dm_drive_type_analysis` (
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `drive_type` VARCHAR(50) NOT NULL COMMENT '驱动形式',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `sales_volume` BIGINT COMMENT '销量',
  `avg_sale_price` DECIMAL(10,2) COMMENT '平均售价',
  `performance_score` DECIMAL(3,1) COMMENT '性能评分(1-10分)',
  `energy_efficiency` DECIMAL(3,1) COMMENT '能源效率(1-10分)',
  PRIMARY KEY (`brand`, `drive_type`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='驱动形式分析表';

-- 13. 充电效率分析表
DROP TABLE IF EXISTS `dm_charging_efficiency_analysis`;

CREATE TABLE `dm_charging_efficiency_analysis` (
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `charging_efficiency_range` VARCHAR(50) NOT NULL COMMENT '充电效率区间',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `sales_volume` BIGINT COMMENT '销量',
  `customer_satisfaction` DECIMAL(3,1) COMMENT '客户满意度(1-10分)',
  `avg_sale_price` DECIMAL(10,2) COMMENT '平均售价',
  `market_acceptance` DECIMAL(3,1) COMMENT '市场接受度(1-10分)',
  PRIMARY KEY (`brand`, `charging_efficiency_range`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='充电效率分析表';

-- 14. 智能化水平分析表
DROP TABLE IF EXISTS `dm_intelligence_level_analysis`;

CREATE TABLE `dm_intelligence_level_analysis` (
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `intelligence_level` VARCHAR(50) NOT NULL COMMENT '智能化水平',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `sales_volume` BIGINT COMMENT '销量',
  `avg_sale_price` DECIMAL(10,2) COMMENT '平均售价',
  `future_potential` DECIMAL(3,1) COMMENT '未来潜力评分(1-10分)',
  PRIMARY KEY (`brand`, `intelligence_level`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='智能化水平分析表';

-- 15. 过户状态分析表
DROP TABLE IF EXISTS `dm_transfer_status_analysis`;

CREATE TABLE `dm_transfer_status_analysis` (
  `brand` VARCHAR(100) NOT NULL COMMENT '电动汽车品牌',
  `transfer_status` VARCHAR(50) NOT NULL COMMENT '过户状态',
  `stat_month` VARCHAR(20) NOT NULL COMMENT '统计月份',
  `transfer_count` BIGINT COMMENT '过户数量',
  `avg_transfer_price` DECIMAL(10,2) COMMENT '平均过户价格',
  `transfer_reason_analysis` VARCHAR(500) COMMENT '过户原因分析',
  PRIMARY KEY (`brand`, `transfer_status`, `stat_month`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='过户状态分析表';