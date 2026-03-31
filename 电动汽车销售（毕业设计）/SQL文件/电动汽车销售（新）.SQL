对MySQL表数据进行空值筛选：

SELECT * FROM electric_car_sales 
WHERE id IS NULL
AND brand IS NULL
AND origin IS NULL
AND production_volume IS NULL
AND sales_volume IS NULL
AND operating_kilometers IS NULL
AND production_date IS NULL
AND member_level IS NULL
AND original_price IS NULL
AND sale_price IS NULL
AND profit IS NULL
AND transfer_status IS NULL
AND manufacturer IS NULL
AND battery_capacity_kwh IS NULL
AND range_nedc_cltc IS NULL
AND charging_efficiency IS NULL
AND drive_type IS NULL
AND market_share IS NULL
AND user_age IS NULL
AND user_gender IS NULL
AND user_occupation IS NULL
AND user_habits IS NULL
AND intelligence_level IS NULL;

筛选后无空值，表数据完整！


在MySQL中创建数据库：
create database electric_car_sales；

使用source命令加载数据：
source /export/data/electric_car_sale/sql/electric_car_sales.sql



--编写JSON文件：electric_car_sales.json--
{
  "job": {
    "content": [
      {
        "reader": {
          "name": "mysqlreader",
          "parameter": {
            "column": [
              "id",
              "brand",
              "origin",
              "production_volume",
              "sales_volume",
              "operating_kilometers",
              "production_date",
              "member_level",
              "original_price",
              "sale_price",
              "profit",
              "transfer_status",
              "manufacturer",
              "battery_capacity_kwh",
              "range_nedc_cltc",
              "charging_efficiency",
              "drive_type",
              "market_share",
              "user_age",
              "user_gender",
              "user_occupation",
              "user_habits",
              "intelligence_level"
            ],
            "connection": [
              {
                "jdbcUrl": [
                  "jdbc:mysql://hadoop01:3306/electric_car_sales?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8"
                ],
                "table": [
                  "electric_car_sales"
                ]
              }
            ],
            "password": "123456",
            "splitPk": "",
            "username": "root"
          }
        },
        "writer": {
          "name": "hdfswriter",
          "parameter": {
            "column": [
              {
                "name": "id",
                "type": "bigint"
              },
              {
                "name": "brand",
                "type": "string"
              },
              {
                "name": "origin",
                "type": "string"
              },
              {
                "name": "production_volume",
                "type": "bigint"
              },
              {
                "name": "sales_volume",
                "type": "bigint"
              },
              {
                "name": "operating_kilometers",
                "type": "string"
              },
              {
                "name": "production_date",
                "type": "string"
              },
              {
                "name": "member_level",
                "type": "string"
              },
              {
                "name": "original_price",
                "type": "string"
              },
              {
                "name": "sale_price",
                "type": "string"
              },
              {
                "name": "profit",
                "type": "string"
              },
              {
                "name": "transfer_status",
                "type": "string"
              },
              {
                "name": "manufacturer",
                "type": "string"
              },
              {
                "name": "battery_capacity_kwh",
                "type": "string"
              },
              {
                "name": "range_nedc_cltc",
                "type": "string"
              },
              {
                "name": "charging_efficiency",
                "type": "string"
              },
              {
                "name": "drive_type",
                "type": "string"
              },
              {
                "name": "market_share",
                "type": "string"
              },
              {
                "name": "user_age",
                "type": "bigint"
              },
              {
                "name": "user_gender",
                "type": "string"
              },
              {
                "name": "user_occupation",
                "type": "string"
              },
              {
                "name": "user_habits",
                "type": "string"
              },
              {
                "name": "intelligence_level",
                "type": "string"
              }
            ],
            "compress": "gzip",
            "defaultFS": "hdfs://hadoop01:9000",
            "fieldDelimiter": "\t",
            "fileName": "electric_car_sales",
            "fileType": "text",
            "path": "/electric_car_sales/car_sales",
            "writeMode": "truncate",
            "nullFormat": ""
          }
        }
      }
    ],
    "setting": {
      "speed": {
        "channel": 1
      }
    }
  }
}


DATAX导入HDFS的SHELL脚本：
#!/bin/bash
DATAX_HOME=/export/servers/datax
DATAX_DATA=/export/servers/datax/job/electric_car_sale/import

handle_targetdir(){
  hadoop fs -rm -r $1 >/dev/null 2>&1
  hadoop fs -mkdir -p $1
}

#数据同步
import_data(){
  target_dir=$1
  datax_config=$2
  handle_targetdir $target_dir
  echo "导入表格数据$datax_config"
  python ${DATAX_HOME}/bin/datax.py -p"-Dtargetdir=$target_dir" $datax_config >/tmp/datax_run.log 2>&1
  if [[ $? -ne 0 ]]; then
    echo "导入数据失败 日志如下"
    cat /tmp/datax_run.log
  fi
}

case $1 in
  "electric_car_sales")
    import_data "/electric_car_sales/car_sales" "$DATAX_DATA/electric_car_sales.json"
    ;;
  "all")
    import_data "/electric_car_sales/car_sales" "$DATAX_DATA/electric_car_sales.json"
    ;;
  *)
    echo "Usage: $0 {electric_car_sales|all}"
    exit 1
esac


赋予全量数据同步脚本赋予权限，执行脚本
chmod +x electric_car_sales_mysql_to_hdfs.sh

electric_car_sales_mysql_to_hdfs.sh all


Hive建表：
数据库
create database electric_car_sale;



原始表：
DROP TABLE IF EXISTS electric_car_sales;

CREATE TABLE electric_car_sales (
  id bigint COMMENT '唯一标识ID',
  brand string COMMENT '电动汽车品牌',
  origin string COMMENT '电动汽车产地',
  production_volume bigint COMMENT '电动汽车产量',
  sales_volume bigint COMMENT '电动汽车销量',
  operating_kilometers string COMMENT '电动汽车运行公里数',  
  production_date string COMMENT '电动汽车生产日期',
  member_level string COMMENT '拥有者会员级别',
  original_price string COMMENT '电动汽车原价',  
  sale_price string COMMENT '电动汽车售价',  
  profit string COMMENT '电动汽车利润',  
  transfer_status string COMMENT '电动汽车过户状态',
  manufacturer string COMMENT '电动汽车厂商',
  battery_capacity_kwh string COMMENT '电池容量(kWh)',  
  range_nedc_cltc string COMMENT '续航里程(NEDC/CLTC)',  
  charging_efficiency string COMMENT '充电效率',
  drive_type string COMMENT '驱动形式',
  market_share string COMMENT '市场份额',  
  user_age bigint COMMENT '用户年龄',
  user_gender string COMMENT '用户性别',
  user_occupation string COMMENT '用户职业',
  user_habits string COMMENT '用户使用习惯',
  intelligence_level string COMMENT '电动汽车智能化水平'
)
COMMENT '电动汽车销售数据表'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'  
STORED AS TEXTFILE  
TBLPROPERTIES (
  'serialization.null.format'='',  
  'mapreduce.output.fileoutputformat.compress'='true',  
  'mapreduce.output.fileoutputformat.compress.codec'='org.apache.hadoop.io.compress.GzipCodec'  
);



hive load data命令进行导入；
load data inpath "/electric_car_sales/car_sales" overwrite into table electric_car_sales;


用SParkSql做hive数据分析：
--项目路径：D:\Masu_BiYe_sheji
--IDEA代码这里不做展示，代码存储在SParkSql.sql文件中
DWD 明细层开发（dwd_electric_car_info）：
清洗 ODS 层原始数据，修正字段类型（字符串转数值、日期格式化）、过滤脏数据，保留明细数据，为后续汇总做基础；

数据库：dwd_electric_car

DWS 汇总层开发（dws_electric_car_agg）：
做轻度汇总（月度 / 地区 / 产品特征的基础聚合），为 DM 层提供聚合后的数据，减少 DM 层计算量；

数据库：dws_electric_car

DM 指标层开发（dm_electric_car_analysis）
基于 DWS 层做深度聚合 + 业务指标计算（环比增长率、排名、占比、评分等），生成最终的 15 张业务指标表；

数据库：dm_electric_car_analysis






Hive指标数据表：
-- 电动汽车销售数据Hive指标表创建脚本
基于原始electric_car_sales表创建15个分析指标表：
1. 月度销售趋势表：dm_monthly_sales_trend，2. 季度销售趋势表（包含环比计算）：dm_quarterly_sales_trend， 3. 年度销售趋势表（包含环比计算）：dm_yearly_sales_trend，
4. 地区销售分析表：dm_region_sales_analysis，5. 客户年龄分析表：dm_customer_age_analysis，6. 客户性别分析表：dm_customer_gender_analysis，
7. 客户职业分析表：dm_customer_occupation_analysis，8. 价格区间分析表：dm_price_range_analysis，9. 利润分析表：dm_profit_analysis，
10. 产品特征分析表-电池容量：dm_battery_capacity_analysis，11. 续航里程分析表：dm_range_analysis，12. 驱动形式分析表：dm_drive_type_analysis，
13. 充电效率分析表：dm_charging_efficiency_analysis，14. 智能化水平分析表：dm_intelligence_level_analysis，15. 过户状态分析表：dm_transfer_status_analysis








MySQL建表：
创建数据库：
CREATE database electric_car_sales_report；

建立导出指标表：

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




编写导出JSON文件，路径：/export/servers/datax/job/electric_car_sale/export

1、月度销售趋势：vi dm_monthly_sales_trend.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_monthly_sales_trend/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "stat_month",
              "brand",
              "total_sales_volume",
              "total_production_volume",
              "avg_sale_price",
              "total_profit"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_monthly_sales_trend;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_monthly_sales_trend "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


2、季度销售趋势：vi dm_quarterly_sales_trend.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_quarterly_sales_trend/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "stat_quarter",
              "brand",
              "total_sales_volume",
              "total_production_volume",
              "avg_sale_price",
              "total_profit"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_quarterly_sales_trend;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_quarterly_sales_trend "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


3、年度销售趋势表：vi dm_yearly_sales_trend.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_yearly_sales_trend/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "stat_year",
              "brand",
              "total_sales_volume",
              "total_production_volume",
              "avg_sale_price",
              "total_profit"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_yearly_sales_trend;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_yearly_sales_trend "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


4、地区销售分析：vi dm_region_sales_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_region_sales_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "origin",
              "brand",
              "stat_month",
              "total_sales_volume",
              "total_production_volume",
              "avg_sale_price",
              "market_share",
              "sales_rank"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_region_sales_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_region_sales_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


5、客户年龄分析：vi dm_customer_age_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_customer_age_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "replace",
            "username": "root",
            "password": "123456",
            "column": [
              "age_group",
              "stat_month",
              "customer_count",
              "total_sales_volume",
              "avg_purchase_price",
              "preferred_brand",
              "preferred_drive_type"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_customer_age_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_customer_age_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


6、客户性别分析：vi dm_customer_gender_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_customer_gender_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "user_gender",
              "stat_month",
              "customer_count",
              "total_sales_volume",
              "avg_purchase_price",
              "preferred_brand"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_customer_gender_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_customer_gender_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


7、客户职业分析：vi dm_customer_occupation_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_customer_occupation_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "user_occupation",
              "stat_month",
              "customer_count",
              "total_sales_volume",
              "avg_purchase_price",
              "preferred_brand",
              "purchase_ability_score"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_customer_occupation_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_customer_occupation_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


8、价格区间分析：vi dm_price_range_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_price_range_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "brand",
              "price_range",
              "stat_month",
              "sales_volume",
              "avg_profit_margin",
              "total_profit",
              "customer_satisfaction_score"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_price_range_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_price_range_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


9、利润分析：vi dm_profit_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_profit_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "brand",
              "stat_month",
              "total_revenue",
              "total_cost",
              "total_profit",
              "profit_margin",
              "profit_per_unit"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_profit_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_profit_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


10、产品特征分析-电池容量：vi dm_battery_capacity_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_battery_capacity_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "brand",
              "battery_range",
              "stat_month",
              "sales_volume",
              "avg_range_nedc",
              "avg_sale_price",
              "customer_preference_score"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_battery_capacity_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_battery_capacity_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


11、续航里程分析：vi dm_range_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_range_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "brand",
              "range_category",
              "stat_month",
              "sales_volume",
              "avg_battery_capacity",
              "avg_sale_price",
              "customer_satisfaction",
              "charging_frequency"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_range_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_range_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


12、驱动形式分析：vi dm_drive_type_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_drive_type_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "brand",
              "drive_type",
              "stat_month",
              "sales_volume",
              "avg_sale_price",
              "performance_score",
              "energy_efficiency"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_drive_type_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_drive_type_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


13、充电效率分析：vi dm_charging_efficiency_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_charging_efficiency_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "brand",
              "charging_efficiency_range",
              "stat_month",
              "sales_volume",
              "customer_satisfaction",
              "avg_sale_price",
              "market_acceptance"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_charging_efficiency_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_charging_efficiency_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}


14、智能化水平分析：vi dm_intelligence_level_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_intelligence_level_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "brand",
              "intelligence_level",
              "stat_month",
              "sales_volume",
              "avg_sale_price",
              "future_potential"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_intelligence_level_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_intelligence_level_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}




15、过户状态分析：vi dm_transfer_status_analysis.json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 1
      }
    },
    "content": [
      {
        "reader": {
          "name": "hdfsreader",
          "parameter": {
            "path": "/user/hive/warehouse/dm_electric_car.db/dm_transfer_status_analysis/*",
            "defaultFS": "hdfs://hadoop01:9000",
            "column": [
              "*"
            ],
            "fileType": "text",
            "encoding": "UTF-8",
            "fieldDelimiter": "\t",
            "nullFormat": "\\N"
          }
        },
        "writer": {
          "name": "mysqlwriter",
          "parameter": {
            "writeMode": "insert",
            "username": "root",
            "password": "123456",
            "column": [
              "brand",
              "transfer_status",
              "stat_month",
              "transfer_count",
              "avg_transfer_price",
              "transfer_reason_analysis"
            ],
            "preSql": [
              "TRUNCATE TABLE dm_transfer_status_analysis;"
            ],
            "connection": [
              {
                "jdbcUrl": "jdbc:mysql://hadoop01:3306/electric_car_sales_report?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf-8",
                "table": [
                  "dm_transfer_status_analysis "
                ]
              }
            ]
          }
        }
      }
    ]
  }
}






在/root/bin目录下创建一键导出脚本：
cd /root/bin

vi electric_car_sale_hdfs_to_mysql.sh

#! /bin/bash

DATAX_HOME=/export/servers/datax
DATAX_DATA=/export/servers/datax/job/electric_car_sale

#DataX导出路径不允许存在空文件，该函数作用为清理空文件
handle_export_path(){
  for i in `hadoop fs -ls -R $1 | awk '{print $8}'`; do
    hadoop fs -test -z $i
    if [[ $? -eq 0 ]]; then
      echo "$i文件大小为0，正在删除"
      hadoop fs -rm -r -f $i
    fi
  done
}

#数据导出
export_data() {
  export_dir=$1
  datax_config=$2
  echo "正在处理 $2 ......"
  handle_export_path $export_dir
  $DATAX_HOME/bin/datax.py -p"-Dexportdir=$export_dir" $datax_config >/tmp/datax_run.log 2>&1
  if [ $? -ne 0 ]
        then
          echo "处理失败，日志如下："
          cat /tmp/datax_run.log
  fi
}

case $1 in
  dm_monthly_sales_trend | dm_quarterly_sales_trend | dm_yearly_sales_trend | dm_region_sales_analysis | dm_customer_age_analysis | dm_customer_gender_analysis | dm_customer_occupation_analysis | dm_price_range_analysis | dm_profit_analysis | dm_battery_capacity_analysis | dm_range_analysis | dm_drive_type_analysis | dm_charging_efficiency_analysis | dm_intelligence_level_analysis | dm_transfer_status_analysis)
    export_data /user/hive/warehouse/dm_electric_car.db/${1} ${DATAX_DATA}/export/$1.json
  ;;

  "all")
    for tab in dm_monthly_sales_trend dm_quarterly_sales_trend dm_yearly_sales_trend dm_region_sales_analysis dm_customer_age_analysis dm_customer_gender_analysis dm_customer_occupation_analysis dm_price_range_analysis dm_profit_analysis dm_battery_capacity_analysis dm_range_analysis dm_drive_type_analysis dm_charging_efficiency_analysis dm_intelligence_level_analysis dm_transfer_status_analysis
    do
      export_data /user/hive/warehouse/dm_electric_car.db/${tab} ${DATAX_DATA}/export/${tab}.json
    done
    ;;
esac


赋予修改权限：
chmod +x electric_car_sale_hdfs_to_mysql.sh

执行脚本：
electric_car_sale_hdfs_to_mysql.sh all

