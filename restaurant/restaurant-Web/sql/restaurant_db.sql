-- ============================================================
-- 火锅到家管理系统 数据库建表脚本
-- 严格按照需求文档 v2.0 设计
-- 适用：MySQL 8.0+
-- ============================================================

CREATE DATABASE IF NOT EXISTS `restaurant_db` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE `restaurant_db`;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ============================================================
-- 一、建表（13张业务表）
-- ============================================================

-- ----------------------------
-- 1. 系统用户表（管理员/店长/店员）
-- ----------------------------
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL COMMENT '用户名',
  `password` varchar(100) NOT NULL COMMENT '密码（BCrypt加密）',
  `real_name` varchar(50) DEFAULT NULL COMMENT '真实姓名',
  `phone` varchar(20) DEFAULT NULL COMMENT '手机号',
  `role` varchar(20) NOT NULL DEFAULT 'STAFF' COMMENT '角色：ADMIN/MANAGER/STAFF',
  `store_id` bigint DEFAULT NULL COMMENT '所属门店ID',
  `avatar` varchar(255) DEFAULT NULL COMMENT '头像URL',
  `status` tinyint NOT NULL DEFAULT 1 COMMENT '状态：0禁用 1启用',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='系统用户表';

-- ----------------------------
-- 2. 顾客用户表
-- ----------------------------
DROP TABLE IF EXISTS `customer`;
CREATE TABLE `customer` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `phone` varchar(20) NOT NULL COMMENT '手机号',
  `password` varchar(100) NOT NULL COMMENT '密码（BCrypt加密）',
  `nickname` varchar(100) DEFAULT NULL COMMENT '昵称',
  `avatar` varchar(255) DEFAULT NULL COMMENT '头像URL',
  `gender` tinyint DEFAULT 0 COMMENT '性别：0未知 1男 2女',
  `status` tinyint NOT NULL DEFAULT 1 COMMENT '状态：0禁用 1正常',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_phone` (`phone`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='顾客用户表';

-- ----------------------------
-- 3. 门店表
-- ----------------------------
DROP TABLE IF EXISTS `store`;
CREATE TABLE `store` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL COMMENT '门店名称',
  `address` varchar(255) DEFAULT NULL COMMENT '门店地址',
  `phone` varchar(20) DEFAULT NULL COMMENT '联系电话',
  `longitude` decimal(10,7) DEFAULT NULL COMMENT '经度',
  `latitude` decimal(10,7) DEFAULT NULL COMMENT '纬度',
  `business_hours` varchar(100) DEFAULT NULL COMMENT '营业时间 如 10:00-22:00',
  `avg_price` decimal(8,2) DEFAULT NULL COMMENT '人均消费',
  `cover_image` varchar(500) DEFAULT NULL COMMENT '封面图',
  `images` text DEFAULT NULL COMMENT '门店图片（JSON数组）',
  `status` tinyint NOT NULL DEFAULT 1 COMMENT '状态：0停业 1营业',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='门店表';

-- ----------------------------
-- 4. 桌型配置表
-- ----------------------------
DROP TABLE IF EXISTS `table_type`;
CREATE TABLE `table_type` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL COMMENT '桌型名称（如：2人桌、4人桌、6人桌、大包间）',
  `capacity_min` int NOT NULL COMMENT '最少容纳人数',
  `capacity_max` int NOT NULL COMMENT '最多容纳人数',
  `store_id` bigint DEFAULT NULL COMMENT '所属门店ID（NULL表示全局桌型）',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='桌型配置表';

-- ----------------------------
-- 5. 门店桌位实例表
-- ----------------------------
DROP TABLE IF EXISTS `store_table`;
CREATE TABLE `store_table` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `store_id` bigint NOT NULL COMMENT '门店ID',
  `table_type_id` bigint NOT NULL COMMENT '桌型ID',
  `table_no` varchar(20) NOT NULL COMMENT '桌号',
  `status` tinyint NOT NULL DEFAULT 1 COMMENT '状态：0停用 1可用',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_store_id` (`store_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='门店桌位实例表';

-- ----------------------------
-- 6. 可预订时段配置表
-- ----------------------------
DROP TABLE IF EXISTS `time_slot`;
CREATE TABLE `time_slot` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `store_id` bigint NOT NULL COMMENT '门店ID',
  `period` varchar(20) NOT NULL COMMENT '时段类型：LUNCH/DINNER',
  `start_time` varchar(10) NOT NULL COMMENT '开始时间 如 11:00',
  `end_time` varchar(10) NOT NULL COMMENT '结束时间 如 14:00',
  `max_tables` int NOT NULL DEFAULT 10 COMMENT '该时段最大可预订桌数',
  `status` tinyint NOT NULL DEFAULT 1 COMMENT '状态：0停用 1启用',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_store_id` (`store_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='可预订时段配置表';

-- ----------------------------
-- 7. 菜品分类表
-- ----------------------------
DROP TABLE IF EXISTS `category`;
CREATE TABLE `category` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL COMMENT '分类名称',
  `sort_order` int NOT NULL DEFAULT 0 COMMENT '排序值（越小越靠前）',
  `store_id` bigint DEFAULT NULL COMMENT '所属门店ID（NULL表示全局分类）',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='菜品分类表';

-- ----------------------------
-- 8. 菜品表
-- ----------------------------
DROP TABLE IF EXISTS `dish`;
CREATE TABLE `dish` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL COMMENT '菜品名称',
  `category_id` bigint NOT NULL COMMENT '分类ID',
  `store_id` bigint DEFAULT NULL COMMENT '所属门店ID（NULL表示全局菜品）',
  `price` decimal(8,2) NOT NULL COMMENT '价格',
  `image` varchar(500) DEFAULT NULL COMMENT '图片URL',
  `description` text DEFAULT NULL COMMENT '描述',
  `spicy_level` tinyint NOT NULL DEFAULT 0 COMMENT '辣度：0-不辣 1-微辣 2-中辣 3-特辣',
  `is_recommended` tinyint NOT NULL DEFAULT 0 COMMENT '是否推荐：0否 1是',
  `is_on_sale` tinyint NOT NULL DEFAULT 1 COMMENT '是否上架：0下架 1上架',
  `sort_order` int NOT NULL DEFAULT 0 COMMENT '排序值',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_category_id` (`category_id`),
  KEY `idx_store_id` (`store_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='菜品表';

-- ----------------------------
-- 9. 预订订单表
-- ----------------------------
DROP TABLE IF EXISTS `reservation`;
CREATE TABLE `reservation` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `order_no` varchar(32) NOT NULL COMMENT '订单编号',
  `customer_id` bigint NOT NULL COMMENT '顾客ID',
  `store_id` bigint NOT NULL COMMENT '门店ID',
  `table_type_id` bigint NOT NULL COMMENT '桌型ID',
  `reserve_date` date NOT NULL COMMENT '预订日期',
  `time_slot_id` bigint NOT NULL COMMENT '时段ID',
  `guest_count` int NOT NULL COMMENT '就餐人数',
  `customer_name` varchar(50) NOT NULL COMMENT '预订人姓名',
  `customer_phone` varchar(20) NOT NULL COMMENT '预订人手机',
  `dish_id` bigint DEFAULT NULL COMMENT '预订套餐ID',
  `remark` varchar(500) DEFAULT NULL COMMENT '备注',
  `status` tinyint NOT NULL DEFAULT 0 COMMENT '状态：0待确认 1已确认 2已到店 3已完成 4已取消 5已拒绝 6未到店',
  `cancel_reason` varchar(255) DEFAULT NULL COMMENT '取消原因',
  `confirm_time` datetime DEFAULT NULL COMMENT '确认时间',
  `arrive_time` datetime DEFAULT NULL COMMENT '到店时间',
  `complete_time` datetime DEFAULT NULL COMMENT '完成时间',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_order_no` (`order_no`),
  KEY `idx_customer_id` (`customer_id`),
  KEY `idx_store_id` (`store_id`),
  KEY `idx_reserve_date` (`reserve_date`),
  KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='预订订单表';

-- ----------------------------
-- 10. 顾客评价表
-- ----------------------------
DROP TABLE IF EXISTS `review`;
CREATE TABLE `review` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `reservation_id` bigint NOT NULL COMMENT '关联的预订订单ID',
  `customer_id` bigint NOT NULL COMMENT '顾客ID',
  `store_id` bigint NOT NULL COMMENT '门店ID',
  `rating` tinyint NOT NULL COMMENT '评分：1-5星',
  `content` varchar(1000) DEFAULT NULL COMMENT '评价内容（文字评价）',
  `images` text DEFAULT NULL COMMENT '评价图片（JSON数组，可选）',
  `is_visible` tinyint NOT NULL DEFAULT 1 COMMENT '是否可见：0隐藏 1显示',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_reservation_id` (`reservation_id`),
  KEY `idx_store_id` (`store_id`),
  KEY `idx_customer_id` (`customer_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='顾客评价表';

-- ----------------------------
-- 11. 轮播图表
-- ----------------------------
DROP TABLE IF EXISTS `banner`;
CREATE TABLE `banner` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `title` varchar(100) DEFAULT NULL COMMENT '标题',
  `image` varchar(500) NOT NULL COMMENT '图片地址',
  `sort_order` int NOT NULL DEFAULT 0 COMMENT '排序值',
  `status` tinyint NOT NULL DEFAULT 1 COMMENT '状态：0隐藏 1显示',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='轮播图表';

-- ----------------------------
-- 12. 门店公告表
-- ----------------------------
DROP TABLE IF EXISTS `announcement`;
CREATE TABLE `announcement` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `store_id` bigint NOT NULL COMMENT '门店ID',
  `title` varchar(100) NOT NULL COMMENT '公告标题',
  `content` text NOT NULL COMMENT '公告内容',
  `status` tinyint NOT NULL DEFAULT 1 COMMENT '状态：0隐藏 1显示',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_store_id` (`store_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='门店公告表';

-- ----------------------------
-- 13. 订单明细表（点餐订单用，二期预留）
-- ----------------------------
DROP TABLE IF EXISTS `order_item`;
CREATE TABLE `order_item` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `order_id` bigint NOT NULL COMMENT '订单ID',
  `dish_id` bigint NOT NULL COMMENT '菜品ID',
  `dish_name` varchar(100) NOT NULL COMMENT '菜品名称（冗余快照）',
  `dish_image` varchar(500) DEFAULT NULL COMMENT '菜品图片（冗余快照）',
  `price` decimal(8,2) NOT NULL COMMENT '单价（下单时快照）',
  `quantity` int NOT NULL COMMENT '数量',
  `deleted` tinyint NOT NULL DEFAULT 0 COMMENT '逻辑删除：0未删除 1已删除',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_order_id` (`order_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单明细表（二期预留）';

SET FOREIGN_KEY_CHECKS = 1;

-- ============================================================
-- 二、初始数据
-- ============================================================

-- ----------------------------
-- 系统用户（密码均为明文 123456 的 BCrypt 加密值）
-- BCrypt hash for "123456": $2a$10$y9faWxiE2.dsMpFoBzrmBu6lIlaVg7DgR9pTf2xYgfPmiFE84Z.ca
-- ----------------------------
INSERT INTO `sys_user` (`username`, `password`, `real_name`, `phone`, `role`, `store_id`, `status`) VALUES
('admin',   '$2a$10$y9faWxiE2.dsMpFoBzrmBu6lIlaVg7DgR9pTf2xYgfPmiFE84Z.ca', '系统管理员', '13800000001', 'ADMIN',   NULL, 1),
('manager', '$2a$10$y9faWxiE2.dsMpFoBzrmBu6lIlaVg7DgR9pTf2xYgfPmiFE84Z.ca', '张店长',   '13800000002', 'MANAGER', 1,    1),
('staff1',  '$2a$10$y9faWxiE2.dsMpFoBzrmBu6lIlaVg7DgR9pTf2xYgfPmiFE84Z.ca', '李店员',   '13800000003', 'STAFF',   1,    1);

-- ----------------------------
-- 顾客用户（密码均为 123456）
-- ----------------------------
INSERT INTO `customer` (`phone`, `password`, `nickname`, `gender`, `status`) VALUES
('13900001111', '$2a$10$y9faWxiE2.dsMpFoBzrmBu6lIlaVg7DgR9pTf2xYgfPmiFE84Z.ca', '张三', 1, 1),
('13900002222', '$2a$10$y9faWxiE2.dsMpFoBzrmBu6lIlaVg7DgR9pTf2xYgfPmiFE84Z.ca', '李四', 2, 1),
('13900003333', '$2a$10$y9faWxiE2.dsMpFoBzrmBu6lIlaVg7DgR9pTf2xYgfPmiFE84Z.ca', '王五', 1, 1);

-- ----------------------------
-- 门店（测试站先做单门店，预留多门店扩展）
-- ----------------------------
INSERT INTO `store` (`name`, `address`, `phone`, `longitude`, `latitude`, `business_hours`, `avg_price`, `cover_image`, `images`, `status`) VALUES
('火锅到家-旗舰店', '北京市朝阳区建国路88号', '010-88886666', 116.4617000, 39.9098000, '10:00-22:00', 128.00, '/images/stores/cover1.jpg', '["/images/stores/env1.jpg","/images/stores/env2.jpg","/images/stores/env3.jpg"]', 1);

-- ----------------------------
-- 桌型配置
-- ----------------------------
INSERT INTO `table_type` (`name`, `capacity_min`, `capacity_max`, `store_id`) VALUES
('2人桌', 1, 2, 1),
('4人桌', 3, 4, 1),
('6人桌', 5, 6, 1),
('8人桌', 7, 8, 1),
('大包间', 8, 12, 1);

-- ----------------------------
-- 门店桌位实例
-- ----------------------------
INSERT INTO `store_table` (`store_id`, `table_type_id`, `table_no`, `status`) VALUES
(1, 1, 'A01', 1), (1, 1, 'A02', 1), (1, 1, 'A03', 1), (1, 1, 'A04', 1), (1, 1, 'A05', 1),
(1, 2, 'B01', 1), (1, 2, 'B02', 1), (1, 2, 'B03', 1), (1, 2, 'B04', 1), (1, 2, 'B05', 1), (1, 2, 'B06', 1), (1, 2, 'B07', 1), (1, 2, 'B08', 1),
(1, 3, 'C01', 1), (1, 3, 'C02', 1), (1, 3, 'C03', 1), (1, 3, 'C04', 1), (1, 3, 'C05', 1),
(1, 4, 'D01', 1), (1, 4, 'D02', 1), (1, 4, 'D03', 1),
(1, 5, 'VIP01', 1), (1, 5, 'VIP02', 1);

-- ----------------------------
-- 可预订时段配置
-- ----------------------------
INSERT INTO `time_slot` (`store_id`, `period`, `start_time`, `end_time`, `max_tables`, `status`) VALUES
(1, 'LUNCH', '11:00', '11:30', 20, 1),
(1, 'LUNCH', '11:30', '12:00', 20, 1),
(1, 'LUNCH', '12:00', '12:30', 20, 1),
(1, 'LUNCH', '12:30', '13:00', 15, 1),
(1, 'LUNCH', '13:00', '13:30', 10, 1),
(1, 'DINNER', '17:00', '17:30', 20, 1),
(1, 'DINNER', '17:30', '18:00', 20, 1),
(1, 'DINNER', '18:00', '18:30', 23, 1),
(1, 'DINNER', '18:30', '19:00', 23, 1),
(1, 'DINNER', '19:00', '19:30', 23, 1),
(1, 'DINNER', '19:30', '20:00', 20, 1),
(1, 'DINNER', '20:00', '20:30', 15, 1),
(1, 'DINNER', '20:30', '21:00', 10, 1);

-- ----------------------------
-- 菜品分类（全局分类，store_id 为 NULL）
-- ----------------------------
INSERT INTO `category` (`name`, `sort_order`, `store_id`) VALUES
('锅底', 1, NULL),
('荤菜', 2, NULL),
('素菜', 3, NULL),
('主食', 4, NULL),
('饮品', 5, NULL),
('小吃', 6, NULL);

-- ----------------------------
-- 菜品（示例数据）
-- ----------------------------
INSERT INTO `dish` (`name`, `category_id`, `store_id`, `price`, `image`, `description`, `spicy_level`, `is_recommended`, `is_on_sale`, `sort_order`) VALUES
-- 锅底
('番茄锅底',   1, NULL, 48.00, '/images/dishes/fanqie.jpg',     '酸甜可口，适合不吃辣的朋友',        0, 1, 1, 1),
('麻辣锅底',   1, NULL, 58.00, '/images/dishes/mala.jpg',       '正宗重庆麻辣风味，鲜香麻辣',        3, 1, 1, 2),
('菌汤锅底',   1, NULL, 52.00, '/images/dishes/juntang.jpg',    '精选多种野生菌熬制，鲜美滋补',      0, 0, 1, 3),
('清汤锅底',   1, NULL, 38.00, '/images/dishes/qingtan.jpg',    '清淡养生，保留食材原味',            0, 0, 1, 4),
('鸳鸯锅底',   1, NULL, 68.00, '/images/dishes/yuanyang.jpg',   '一半麻辣一半清汤，满足不同口味',    2, 1, 1, 5),
-- 荤菜
('精品肥牛',   2, NULL, 48.00, '/images/dishes/feiniu.jpg',     '澳洲进口肥牛卷，肉质鲜嫩',          0, 1, 1, 1),
('精品羊肉',   2, NULL, 48.00, '/images/dishes/yangrou.jpg',    '内蒙锡盟羊肉，不膻不腻',            0, 1, 1, 2),
('毛肚',      2, NULL, 38.00, '/images/dishes/maodu.jpg',       '新鲜毛肚，七上八下涮出脆嫩口感',    0, 1, 1, 3),
('鸭肠',      2, NULL, 32.00, '/images/dishes/yachang.jpg',     '脆爽鸭肠，火锅必点',                0, 0, 1, 4),
('黄喉',      2, NULL, 28.00, '/images/dishes/huanghou.jpg',    '爽脆黄喉，涮10秒即可',              0, 0, 1, 5),
('虾滑',      2, NULL, 38.00, '/images/dishes/xiahua.jpg',      '手打鲜虾滑，Q弹爽滑',               0, 1, 1, 6),
('牛肉丸',    2, NULL, 32.00, '/images/dishes/niurouwan.jpg',   '潮汕手打牛肉丸，弹牙爽口',          0, 0, 1, 7),
('午餐肉',    2, NULL, 22.00, '/images/dishes/wucanrou.jpg',    '经典午餐肉，厚切大片',              0, 0, 1, 8),
-- 素菜
('大白菜',    3, NULL, 12.00, '/images/dishes/baicai.jpg',      '新鲜大白菜，火锅百搭',              0, 0, 1, 1),
('生菜',      3, NULL, 12.00, '/images/dishes/shengcai.jpg',    '脆嫩生菜，涮一下就吃',              0, 0, 1, 2),
('金针菇',    3, NULL, 15.00, '/images/dishes/jinzhen.jpg',     '火锅必备素菜',                      0, 0, 1, 3),
('香菇',      3, NULL, 18.00, '/images/dishes/xianggu.jpg',     '鲜香菇，吸满锅底鲜味',              0, 0, 1, 4),
('土豆片',    3, NULL, 12.00, '/images/dishes/tudou.jpg',       '现切土豆片，煮软糯好吃',            0, 0, 1, 5),
('藕片',      3, NULL, 15.00, '/images/dishes/oupian.jpg',      '脆藕片，涮出清脆口感',              0, 0, 1, 6),
('豆腐',      3, NULL, 12.00, '/images/dishes/doufu.jpg',       '嫩豆腐，入口即化',                  0, 0, 1, 7),
('冻豆腐',    3, NULL, 15.00, '/images/dishes/dongdoufu.jpg',   '蜂窝冻豆腐，吸满汤汁超满足',        0, 0, 1, 8),
-- 主食
('手擀面',    4, NULL, 15.00, '/images/dishes/shougansmian.jpg','手工擀制，筋道爽滑',                0, 0, 1, 1),
('宽粉',      4, NULL, 12.00, '/images/dishes/kuanfen.jpg',     '红薯宽粉，久煮不烂',                0, 0, 1, 2),
('年糕',      4, NULL, 15.00, '/images/dishes/niangao.jpg',     '软糯年糕，蘸酱好吃',                0, 0, 1, 3),
-- 饮品
('酸梅汤',    5, NULL, 15.00, '/images/dishes/suanmeitang.jpg', '传统酸梅汤，解辣解腻',              0, 1, 1, 1),
('可乐',      5, NULL,  8.00, '/images/dishes/kele.jpg',        '冰镇可乐',                          0, 0, 1, 2),
('雪碧',      5, NULL,  8.00, '/images/dishes/xuebi.jpg',       '冰镇雪碧',                          0, 0, 1, 3),
('王老吉',    5, NULL,  8.00, '/images/dishes/wanglaoji.jpg',   '怕上火喝王老吉',                    0, 0, 1, 4),
-- 小吃
('红糖糍粑',  6, NULL, 22.00, '/images/dishes/ciba.jpg',        '外酥里糯，红糖浇淋',                0, 1, 1, 1),
('酥肉',      6, NULL, 28.00, '/images/dishes/surou.jpg',       '现炸小酥肉，外酥里嫩',              1, 0, 1, 2),
('凉拌毛豆',  6, NULL, 15.00, '/images/dishes/maodou.jpg',      '开胃小菜',                          1, 0, 1, 3);

-- ----------------------------
-- 轮播图
-- ----------------------------
INSERT INTO `banner` (`title`, `image`, `sort_order`, `status`) VALUES
('新用户首单立减20元',  '/images/banners/new-user.jpg',    1, 1),
('双人套餐限时特惠',    '/images/banners/shuangren.jpg',   2, 1),
('冬季暖身火锅节',      '/images/banners/winter-fest.jpg', 3, 1);

-- ----------------------------
-- 门店公告
-- ----------------------------
INSERT INTO `announcement` (`store_id`, `title`, `content`, `status`) VALUES
(1, '新店开业优惠', '即日起至本月底，全场锅底8折，预订即送酸梅汤一扎！', 1),
(1, '温馨提示', '周末高峰期建议提前2天预订，包间需提前3天预订。', 1);

-- ============================================================
-- 三、增量DDL（已有数据库执行以下SQL即可）
-- ============================================================

-- 预订表增加套餐ID字段
ALTER TABLE `reservation` ADD COLUMN `dish_id` bigint DEFAULT NULL COMMENT '预订套餐ID' AFTER `customer_phone`;
