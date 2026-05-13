
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for activity
-- ----------------------------
DROP TABLE IF EXISTS `activity`;
CREATE TABLE `activity`  (
  `id` bigint(0) NOT NULL,
  `archives_type_code` int(0) NULL DEFAULT NULL COMMENT '档案类型',
  `audit_status_code` int(0) NULL DEFAULT NULL COMMENT '档案当前状态字典码',
  `stu_id` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '学号',
  `activity_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '活动/竞赛名称',
  `activity_level` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '活动/竞赛等级',
  `activity_type` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '活动/竞赛类型',
  `activity_time` date NULL DEFAULT NULL COMMENT '参与/获得时间',
  `img` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '参与/获得证明',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of activity
-- ----------------------------
INSERT INTO `activity` VALUES (1, 1, 1, '17101020312', '中国智能魔方机器人大赛', '校级', '校园活动', '2020-08-26', ';/api/upload/images/test.jpg', 0, '2020-08-26 00:00:00', '2020-08-26 00:00:00');
INSERT INTO `activity` VALUES (2, 1, 1, '171010245415', '四川省机器人大赛', '省部级', '学术竞赛', '2020-08-26', '', 0, '2020-08-26 00:00:00', '2020-08-26 00:00:00');

-- ----------------------------
-- Table structure for archive
-- ----------------------------
DROP TABLE IF EXISTS `archive`;
CREATE TABLE `archive`  (
  `id` bigint(0) NOT NULL AUTO_INCREMENT,
  `username` varchar(128) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  `user_id` bigint(0) NOT NULL DEFAULT 1,
  `borr_count` int(0) NOT NULL DEFAULT 0,
  `cut_off_time` datetime(0) NULL DEFAULT NULL,
  `deleted` tinyint(0) NOT NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0),
  `modified` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1299237213129203714 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of archive
-- ----------------------------
INSERT INTO `archive` VALUES (1, '4466', 1298295340520792065, 2, '2020-08-30 00:00:00', 0, '2020-08-26 00:00:00', '2020-08-27 12:02:18');
INSERT INTO `archive` VALUES (2, 'lainhong', 1298295618145968130, 52, '2020-09-25 00:00:00', 0, '2020-08-28 00:00:00', '2020-08-26 23:39:31');
INSERT INTO `archive` VALUES (3, 'ffbnfbnf', 1298295680766926849, 7, '2020-09-18 13:03:40', 0, '2020-08-25 07:29:01', NULL);
INSERT INTO `archive` VALUES (6, 'ffbnfnn', 1298295890977, 2, '2020-08-26 00:00:00', 0, '2020-08-26 00:00:00', '2020-08-26 23:42:55');
INSERT INTO `archive` VALUES (7, '987987', 1298307367641395201, 2, '2020-08-26 00:00:00', 0, '2020-08-26 00:00:00', '2020-08-26 23:40:00');
INSERT INTO `archive` VALUES (1298834583231434754, '埃尔文群二', 312321, 3, '2020-08-28 00:00:00', 0, '2020-08-18 00:00:00', '2020-08-27 12:07:50');
INSERT INTO `archive` VALUES (1299237213129203713, 'test', 17101020319, 2, '2020-08-11 00:00:00', 0, '2020-08-11 00:00:00', '2020-08-28 14:47:45');

-- ----------------------------
-- Table structure for audit
-- ----------------------------
DROP TABLE IF EXISTS `audit`;
CREATE TABLE `audit`  (
  `id` bigint(0) NOT NULL COMMENT '唯一标识',
  `archive_type_code` bigint(0) NULL DEFAULT NULL COMMENT '对应的档案是哪一个类型',
  `archive_id` bigint(0) NULL DEFAULT NULL COMMENT '档案对应的ID',
  `deleted` tinyint(0) NULL DEFAULT NULL COMMENT '是否删除',
  `created` date NULL DEFAULT NULL,
  `modified` date NULL DEFAULT NULL,
  `stu_id` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '学号',
  `is_audited` tinyint(0) NULL DEFAULT NULL COMMENT '是否审核,0未审核，1通过审核',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of audit
-- ----------------------------
INSERT INTO `audit` VALUES (1299373884390858754, 2, 1299373883644272642, 0, '2020-08-28', '2020-08-29', '17101020312', 1);
INSERT INTO `audit` VALUES (1299595622722641922, 2, 1299595621971861505, 0, '2020-08-29', '2020-08-29', '17101020', 1);
INSERT INTO `audit` VALUES (1299595830927892481, 4, 1299595829342445570, 0, '2020-08-29', '2020-08-29', '17101020', 1);

-- ----------------------------
-- Table structure for course
-- ----------------------------
DROP TABLE IF EXISTS `course`;
CREATE TABLE `course`  (
  `id` bigint(0) NOT NULL,
  `archives_type_code` bigint(0) NULL DEFAULT NULL COMMENT '档案类型',
  `audit_status_code` bigint(0) NULL DEFAULT NULL COMMENT '档案当前状态字典码',
  `stu_id` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '学号',
  `course_name` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '课程名称',
  `score` float NULL DEFAULT NULL COMMENT '课程成绩',
  `course_level` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '课程成绩等级，优良中',
  `course_rank` int(0) NULL DEFAULT NULL COMMENT '排名',
  `teacher` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '任课老师',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  `usual_score` float NULL DEFAULT NULL COMMENT '平时成绩',
  `usual_ratio` float NULL DEFAULT NULL COMMENT '平时成绩占比',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of course
-- ----------------------------
INSERT INTO `course` VALUES (1, 1, 3, '1', '高等数学', 95, '优', 1, '岳建明', 0, '2020-08-26 00:00:00', '2020-08-26 00:00:00', 100, 0.8);
INSERT INTO `course` VALUES (2, 2, 2, '1', 'JAVA程序开发', 99, '良', 1, '梁兴建', 0, '2020-08-26 00:00:00', '2020-08-26 00:00:00', 95, 0.4);
INSERT INTO `course` VALUES (3, 1, 1, '1', '英语', 65, '中', 1, '老师', 0, '2020-08-26 11:40:49', '2020-08-26 11:40:51', 70, 0.6);
INSERT INTO `course` VALUES (1299240213147140098, 1, 1, '17110203333', 'JAVA实训课程开发', 90, '优', NULL, '老师', 0, '2020-08-28 14:59:40', '2020-08-28 14:59:40', 100, 0.4);

-- ----------------------------
-- Table structure for dictionary
-- ----------------------------
DROP TABLE IF EXISTS `dictionary`;
CREATE TABLE `dictionary`  (
  `id` bigint(0) NOT NULL COMMENT '唯一标识',
  `dict_value` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '字典值',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  `dict_key` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '字典key',
  `dict_code` bigint(0) NULL DEFAULT NULL COMMENT '字典码',
  `dict_desc` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '描述',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of dictionary
-- ----------------------------
INSERT INTO `dictionary` VALUES (1, '待审核', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00', 'audit_status', 1, NULL);
INSERT INTO `dictionary` VALUES (2, '审核中', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00', 'audit_status', 2, NULL);
INSERT INTO `dictionary` VALUES (3, '审核通过', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00', 'audit_status', 3, NULL);
INSERT INTO `dictionary` VALUES (4, '档案已入库', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00', 'archive_status', 1, NULL);
INSERT INTO `dictionary` VALUES (5, '被驳回', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00', 'audit_status', 3, NULL);
INSERT INTO `dictionary` VALUES (6, '档案已出库', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00', 'archive_status', 2, NULL);
INSERT INTO `dictionary` VALUES (7, '借阅中', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00', 'archive_status', 3, NULL);
INSERT INTO `dictionary` VALUES (8, 'project', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00', 'archive_type', 1, '科研项目');
INSERT INTO `dictionary` VALUES (9, 'medical', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00', 'archive_type', 2, '体检记录');
INSERT INTO `dictionary` VALUES (10, 'course', 0, '2020-08-27 17:07:55', '2020-08-27 17:07:53', 'archive_type', 3, '课程记录');
INSERT INTO `dictionary` VALUES (11, 'discipline', 0, '2020-08-27 17:09:47', '2020-08-27 17:09:45', 'archive_type', 4, '违纪记录');
INSERT INTO `dictionary` VALUES (12, 'employ', 0, '2020-08-28 23:02:01', '2020-08-28 23:01:58', 'archive_type', 5, '就业记录');

-- ----------------------------
-- Table structure for discipline
-- ----------------------------
DROP TABLE IF EXISTS `discipline`;
CREATE TABLE `discipline`  (
  `id` bigint(0) NOT NULL,
  `archives_type_code` int(0) NULL DEFAULT 4 COMMENT '档案类型',
  `audit_status_code` int(0) NULL DEFAULT 1 COMMENT '档案当前状态字典码',
  `stu_id` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '学号',
  `dis_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '违纪名称',
  `dis_time` date NULL DEFAULT NULL COMMENT '违纪时间',
  `dis_desc` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '违纪描述',
  `measure` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '处理措施',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of discipline
-- ----------------------------
INSERT INTO `discipline` VALUES (1, 1, 1, '1710102454545', '警告处分', '2020-08-27', '毁坏公共物品', '记大过', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00');
INSERT INTO `discipline` VALUES (1299362925089767426, 4, 1, '112121', '2121212', '2020-08-11', '121212', NULL, 0, '2020-08-28 23:07:17', '2020-08-28 23:07:17');
INSERT INTO `discipline` VALUES (1299366714588901378, 4, 1, '17102', '不不不分割', '2020-08-11', '百分比更方便', NULL, 0, '2020-08-28 23:22:20', '2020-08-28 23:22:20');
INSERT INTO `discipline` VALUES (1299366972437823490, 4, 1, '424', '24242', '2020-08-04', '452452', NULL, 0, '2020-08-28 23:23:22', '2020-08-28 23:23:22');
INSERT INTO `discipline` VALUES (1299368054186680321, 4, 1, '5434', '345343', '2020-08-05', '343', NULL, 0, '2020-08-28 23:27:40', '2020-08-28 23:27:40');
INSERT INTO `discipline` VALUES (1299368682736693249, 4, 1, '732737', '373737', '2020-08-05', '73873', NULL, 0, '2020-08-28 23:30:10', '2020-08-28 23:30:10');
INSERT INTO `discipline` VALUES (1299595829342445570, 4, 3, '17101020', '严重违纪', '2020-08-11', '记大过处分；留校查看', NULL, 0, '2020-08-29 14:32:46', '2020-08-29 14:32:46');

-- ----------------------------
-- Table structure for employ
-- ----------------------------
DROP TABLE IF EXISTS `employ`;
CREATE TABLE `employ`  (
  `id` bigint(0) NOT NULL COMMENT '唯一标识',
  `stu_id` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '学号',
  `company` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '毕业后入职的第一家公司名字',
  `in_time` date NULL DEFAULT NULL COMMENT '入职时间',
  `salary` float NULL DEFAULT NULL COMMENT '薪资',
  `address` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '公司地址',
  `deleted` tinyint(0) NULL DEFAULT NULL COMMENT '是否删除',
  `created` date NULL DEFAULT NULL,
  `modified` date NULL DEFAULT NULL,
  `archives_type_code` bigint(0) NULL DEFAULT 5 COMMENT '档案类型',
  `audit_status_code` bigint(0) NULL DEFAULT 1 COMMENT '审核状态',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of employ
-- ----------------------------
INSERT INTO `employ` VALUES (1, '17101054', '四川中电启明星', '2020-07-27', 2300, '四川成都', 0, '2020-08-28', '2020-08-28', 1, 1);
INSERT INTO `employ` VALUES (1299134268375080961, '17101020315646', '阿里巴巴', '2020-08-28', 5000, NULL, 0, '2020-08-28', '2020-08-28', 1, 1);

-- ----------------------------
-- Table structure for medical
-- ----------------------------
DROP TABLE IF EXISTS `medical`;
CREATE TABLE `medical`  (
  `id` bigint(0) NOT NULL,
  `archives_type_code` bigint(0) NULL DEFAULT 2 COMMENT '档案类型',
  `audit_status_code` bigint(0) NULL DEFAULT 1 COMMENT '档案当前状态字典码',
  `stu_id` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '学号',
  `medical_desc` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '体检结果说明',
  `img` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '体检结果图片(多张用分号分割)',
  `medical_time` date NULL DEFAULT NULL COMMENT '体检时间',
  `unit` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '体检单位',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of medical
-- ----------------------------
INSERT INTO `medical` VALUES (1, 1, 1, '17101020', '红细胞：一般情况下测定血红蛋白和红细胞二者的意义基本相同，其减少往往提示有贫血，可见缺铁性贫血、营养不良性贫血、失血性贫血和再生障碍性贫血等。一般标准是：血红蛋白9-12克为轻度贫血、6-9克为中度贫血、3-6克为重度贫血、3克以下为极重度贫血。我们一般人体检中常见的多为轻度贫血，多见是缺铁性贫血。红细胞增多则可见剧烈吐泻、高热大汗等导致脱水的情况下，还有就是长期缺氧的情况下。', ';/api//upload//images/934d6549-9f5a-47e6-b953-fcc66f588e85.jpg', '2020-08-13', '重庆市人民医院', 0, '2020-08-27 00:00:00', '2020-08-27 00:00:00');
INSERT INTO `medical` VALUES (1298867726626451457, 1, 1, '17101020313', '体检未见任何异常', ';/api//upload//images/c88c1663-9f23-4c98-8bb3-50badc301e46.jpg', '2020-08-27', '重庆市人民医院', 0, '2020-08-27 14:19:32', '2020-08-27 14:19:32');
INSERT INTO `medical` VALUES (1299373883644272642, 2, 3, '17101020312', '测试测试大城市的', NULL, '2020-08-12', '656', 0, '2020-08-28 23:50:50', '2020-08-28 23:50:50');
INSERT INTO `medical` VALUES (1299595621971861505, 2, 3, '17101020', '未见任何异常', NULL, '2020-08-26', '重庆市人民医院', 0, '2020-08-29 14:31:56', '2020-08-29 14:31:56');

-- ----------------------------
-- Table structure for permission
-- ----------------------------
DROP TABLE IF EXISTS `permission`;
CREATE TABLE `permission`  (
  `id` bigint(0) NOT NULL,
  `title` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '权限简称',
  `descr` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '权限描述',
  `url` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '权限访问路径',
  `perms` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '进行细细粒度划分时需要的一个标识',
  `parent_id` int(0) NULL DEFAULT NULL COMMENT '父级权限ID',
  `type` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '权限所属类型：菜单控制，按钮控制，页面控制',
  `icon` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '图标地址',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of permission
-- ----------------------------
INSERT INTO `permission` VALUES (1299278105429000193, '系统设置', '可以设置系统属性', NULL, 'menu:setting', NULL, '菜单', NULL, 0, '2020-08-28 00:00:00', '2020-08-28 00:00:00');
INSERT INTO `permission` VALUES (1299285167219261441, '学生基本信息', '管理学生基本信息的菜单', NULL, 'menu:stu_info', NULL, '菜单', NULL, 0, '2020-08-28 17:58:18', '2020-08-28 17:58:18');
INSERT INTO `permission` VALUES (1299285258143383553, '课程项目菜单', NULL, NULL, 'menu:course_project', NULL, '菜单', NULL, 0, '2020-08-28 00:00:00', '2020-08-28 00:00:00');
INSERT INTO `permission` VALUES (1299285406823071746, '活动竞赛', NULL, NULL, 'menu:activity', NULL, '菜单', NULL, 0, '2020-08-28 17:59:15', '2020-08-28 17:59:15');
INSERT INTO `permission` VALUES (1299285486804254722, '健康体检', NULL, NULL, 'menu:medical', NULL, '菜单', NULL, 0, '2020-08-28 17:59:34', '2020-08-28 17:59:34');
INSERT INTO `permission` VALUES (1299285554110251010, '违纪信息', NULL, NULL, 'menu:weijiMes', NULL, '菜单', NULL, 0, '2020-08-28 17:59:50', '2020-08-28 17:59:50');
INSERT INTO `permission` VALUES (1299285616886398977, '就业发展信息', NULL, NULL, 'menu:grow_infro', NULL, '菜单', NULL, 0, '2020-08-28 18:00:05', '2020-08-28 18:00:05');
INSERT INTO `permission` VALUES (1299285668707024897, '档案信息', NULL, NULL, 'menu:archives', NULL, '菜单', NULL, 0, '2020-08-28 18:00:18', '2020-08-28 18:00:18');
INSERT INTO `permission` VALUES (1299285899792203777, '用户管理', NULL, NULL, 'menu:user_manager', NULL, '菜单', NULL, 0, '2020-08-28 18:01:13', '2020-08-28 18:01:13');
INSERT INTO `permission` VALUES (1299285959032553473, '角色管理', NULL, NULL, 'menu:role_manager', NULL, '菜单', NULL, 0, '2020-08-28 18:01:27', '2020-08-28 18:01:27');
INSERT INTO `permission` VALUES (1299286034576162817, '权限管理', NULL, NULL, 'menu:permission_manager', NULL, '菜单', NULL, 0, '2020-08-28 18:01:45', '2020-08-28 18:01:45');
INSERT INTO `permission` VALUES (1299293088007184386, '档案借阅管理', NULL, NULL, 'menu:archives_borrow', NULL, '菜单', NULL, 0, '2020-08-28 18:29:47', '2020-08-28 18:29:47');
INSERT INTO `permission` VALUES (1299298737512333313, '系统管理', NULL, NULL, 'menu:system_manager', NULL, '菜单', NULL, 0, '2020-08-28 18:52:13', '2020-08-28 18:52:13');
INSERT INTO `permission` VALUES (1299581080902901762, '学生基本信息表格', '可以管理学生的基本信息，查看任意一个学生的信息', NULL, 'stuInfoManager', NULL, '资源', NULL, 0, '2020-08-29 13:34:09', '2020-08-29 13:34:09');

-- ----------------------------
-- Table structure for physical_archives
-- ----------------------------
DROP TABLE IF EXISTS `physical_archives`;
CREATE TABLE `physical_archives`  (
  `id` bigint(0) NOT NULL,
  `archives_type_code` int(0) NULL DEFAULT -1 COMMENT '档案类型',
  `audit_status_code` int(0) NULL DEFAULT 1 COMMENT '档案当前状态字典码',
  `stu_id` bigint(0) NULL DEFAULT NULL COMMENT '学号',
  `archive_name` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '档案名称',
  `position` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '存放位置',
  `archive_desc` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '档案说明',
  `borrow_time` int(0) NULL DEFAULT NULL COMMENT '档案允许被借阅分钟数',
  `img` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '档案图片(多张用分号分割)',
  `in_time` date NULL DEFAULT NULL COMMENT '入库时间',
  `operator_id` bigint(0) NULL DEFAULT NULL COMMENT '操作人员ID(如：入库人员)',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of physical_archives
-- ----------------------------
INSERT INTO `physical_archives` VALUES (1, 1, 4, 17101020316, '学习', '学习', '无', 1565, 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1598597692987&di=09b32cca76149645a65ed67fe2592342&imgtype=0&src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20180126%2F8ae3e881d8fc4a61bb4b571d193e81cf.jpeg', '2020-08-28', 5, 0, '2020-08-28 09:54:45', '2020-08-28 09:54:47');
INSERT INTO `physical_archives` VALUES (2, 5, 4, 17101020312, '娱乐', '娱乐', '无', 46848, 'https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=1217155543,1792161616&fm=26&gp=0.jpg', '2020-08-28', 6, 0, '2020-08-27 12:10:05', '2020-08-28 12:10:08');
INSERT INTO `physical_archives` VALUES (3, -1, 6, 1710102046, '生活', '生活', NULL, 14646, 'https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=1217155543,1792161616&fm=26&gp=0.jpg', '2020-08-27', 1, 0, NULL, NULL);
INSERT INTO `physical_archives` VALUES (4, -1, 6, 171010204496, '影视', '影视', NULL, 464, 'https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=1217155543,1792161616&fm=26&gp=0.jpg', '2020-08-29', 3, 0, NULL, NULL);
INSERT INTO `physical_archives` VALUES (5, -1, 4, 1710203459, '事业', '事业', NULL, 4684, 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1598597692987&di=09b32cca76149645a65ed67fe2592342&imgtype=0&src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20180126%2F8ae3e881d8fc4a61bb4b571d193e81cf.jpeg', '2020-08-29', 4, 0, NULL, NULL);
INSERT INTO `physical_archives` VALUES (6, -1, 6, 17102564, '人生', '人生', NULL, 10, 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1598597692987&di=09b32cca76149645a65ed67fe2592342&imgtype=0&src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20180126%2F8ae3e881d8fc4a61bb4b571d193e81cf.jpeg', '2020-08-29', NULL, 0, NULL, NULL);

-- ----------------------------
-- Table structure for physical_borr_record
-- ----------------------------
DROP TABLE IF EXISTS `physical_borr_record`;
CREATE TABLE `physical_borr_record`  (
  `id` bigint(0) NOT NULL,
  `archive_id` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '档案id',
  `archive_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '档案名称',
  `stu_id` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被借阅学生id',
  `stu_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '被借阅学生名',
  `username_id` bigint(0) NULL DEFAULT NULL COMMENT '借阅者id',
  `username` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '借阅者',
  `borr_time` date NULL DEFAULT NULL COMMENT '借阅时间',
  `borr_status` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '借阅状态，1借阅中，0已归还，-1超时',
  `created` date NULL DEFAULT NULL COMMENT '创建时间',
  `modified` date NULL DEFAULT NULL COMMENT '更新时间',
  `deleted` date NULL DEFAULT NULL COMMENT '删除时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of physical_borr_record
-- ----------------------------
INSERT INTO `physical_borr_record` VALUES (1, '54646464654', 'java程序设计', '17101020316', 'lxb', 17101204653456, 'xxx', '2020-08-29', '1', '2020-08-29', NULL, NULL);
INSERT INTO `physical_borr_record` VALUES (2, '46848668464', 'c语言程序设计', '17101020316', 'ewj', 17101204653456, 'dflgfy', '2020-08-29', '0', '2020-08-29', NULL, NULL);
INSERT INTO `physical_borr_record` VALUES (3, '47967676', 'python程序设计', '494644435', 'tuy', 2, '1jhjhj', '2020-08-25', '-1', NULL, NULL, NULL);
INSERT INTO `physical_borr_record` VALUES (4, '5435435435', 'php程序设计', '444343432', 'hhgt', 2, 'gdfgf', '2020-08-15', '1', NULL, NULL, NULL);

-- ----------------------------
-- Table structure for project
-- ----------------------------
DROP TABLE IF EXISTS `project`;
CREATE TABLE `project`  (
  `id` bigint(20) UNSIGNED ZEROFILL NOT NULL,
  `archives_type_code` int(0) NULL DEFAULT NULL COMMENT '档案类型',
  `audit_status_code` int(0) NULL DEFAULT NULL COMMENT '档案当前状态字典码',
  `stu_id` bigint(0) NULL DEFAULT NULL COMMENT '学号',
  `pro_name` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '项目名称',
  `pro_level` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '项目级别',
  `funds` float NULL DEFAULT NULL COMMENT '项目经费',
  `status` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '项目状态',
  `descr` varchar(300) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '项目简介',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of project
-- ----------------------------
INSERT INTO `project` VALUES (01298569678629224449, 1, 1, 171010203, '智慧城市停车系统', '国家级', 10000, '完成中', '智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车系统智慧城市停车', 0, '2020-08-26 18:35:12', '2020-08-26 18:35:12');
INSERT INTO `project` VALUES (01298573106973622273, 1, 1, 121212, 'Fast Mock', '省部级', 0, '完成中', 'Fast Mock是一个快速Mock网站，能够高效的模拟出各种数据。同时使用restFul接口，方便使用。', 0, '2020-08-26 18:48:50', '2020-08-26 18:48:50');
INSERT INTO `project` VALUES (01299240489061040130, 1, 1, 12121, '大创项目', '国家级', 10000, '已立项', '大创项目简介', 0, '2020-08-28 15:00:46', '2020-08-28 15:00:46');

-- ----------------------------
-- Table structure for role
-- ----------------------------
DROP TABLE IF EXISTS `role`;
CREATE TABLE `role`  (
  `id` bigint(0) NOT NULL,
  `role_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '角色名字',
  `role_desc` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '角色描述',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  `role_mark` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '角色标识',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of role
-- ----------------------------
INSERT INTO `role` VALUES (1299194681653555201, '超级管理员', '拥有所有权限', 0, '2020-08-28 00:00:00', '2020-08-28 00:00:00', 'admin');
INSERT INTO `role` VALUES (1299213524224532481, '工作人员', '我是工作人员', 0, '2020-08-28 00:00:00', '2020-08-28 00:00:00', 'work');
INSERT INTO `role` VALUES (1299241863660290049, '医护人员', '医护人员角色', 0, '2020-08-28 00:00:00', '2020-08-28 00:00:00', 'medical');
INSERT INTO `role` VALUES (1299293407076278273, '学生', '普通用户', 0, '2020-08-28 00:00:00', '2020-08-28 00:00:00', 'student');

-- ----------------------------
-- Table structure for role_permission
-- ----------------------------
DROP TABLE IF EXISTS `role_permission`;
CREATE TABLE `role_permission`  (
  `id` bigint(0) NOT NULL,
  `permission_id` bigint(0) NULL DEFAULT NULL,
  `role_id` bigint(0) NULL DEFAULT NULL,
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of role_permission
-- ----------------------------
INSERT INTO `role_permission` VALUES (1299291879930191873, 1299285616886398977, 1299194681653555201, 0, '2020-08-28 18:24:59', '2020-08-28 18:24:59');
INSERT INTO `role_permission` VALUES (1299292333808410626, 1299278105429000193, 1299194681653555201, 0, '2020-08-28 18:26:47', '2020-08-28 18:26:47');
INSERT INTO `role_permission` VALUES (1299292333816799233, 1299285167219261441, 1299194681653555201, 0, '2020-08-28 18:26:47', '2020-08-28 18:26:47');
INSERT INTO `role_permission` VALUES (1299292333825187841, 1299285258143383553, 1299194681653555201, 0, '2020-08-28 18:26:47', '2020-08-28 18:26:47');
INSERT INTO `role_permission` VALUES (1299292333833576449, 1299285406823071746, 1299194681653555201, 0, '2020-08-28 18:26:47', '2020-08-28 18:26:47');
INSERT INTO `role_permission` VALUES (1299292333833576451, 1299285486804254722, 1299194681653555201, 0, '2020-08-28 18:26:47', '2020-08-28 18:26:47');
INSERT INTO `role_permission` VALUES (1299292333841965058, 1299285554110251010, 1299194681653555201, 0, '2020-08-28 18:26:47', '2020-08-28 18:26:47');
INSERT INTO `role_permission` VALUES (1299292333841965060, 1299285668707024897, 1299194681653555201, 0, '2020-08-28 18:26:47', '2020-08-28 18:26:47');
INSERT INTO `role_permission` VALUES (1299292333841965062, 1299285899792203777, 1299194681653555201, 0, '2020-08-28 18:26:47', '2020-08-28 18:26:47');
INSERT INTO `role_permission` VALUES (1299292333850353665, 1299285959032553473, 1299194681653555201, 0, '2020-08-28 18:26:47', '2020-08-28 18:26:47');
INSERT INTO `role_permission` VALUES (1299292333850353667, 1299286034576162817, 1299194681653555201, 0, '2020-08-28 18:26:47', '2020-08-28 18:26:47');
INSERT INTO `role_permission` VALUES (1299293130084442114, 1299293088007184386, 1299194681653555201, 0, '2020-08-28 18:29:57', '2020-08-28 18:29:57');
INSERT INTO `role_permission` VALUES (1299293407487320066, 1299285167219261441, 1299293407076278273, 0, '2020-08-28 18:31:03', '2020-08-28 18:31:03');
INSERT INTO `role_permission` VALUES (1299298042109313025, 1299285258143383553, 1299293407076278273, 0, '2020-08-28 18:49:28', '2020-08-28 18:49:28');
INSERT INTO `role_permission` VALUES (1299298042134478850, 1299285406823071746, 1299293407076278273, 0, '2020-08-28 18:49:28', '2020-08-28 18:49:28');
INSERT INTO `role_permission` VALUES (1299298042142867457, 1299285486804254722, 1299293407076278273, 0, '2020-08-28 18:49:28', '2020-08-28 18:49:28');
INSERT INTO `role_permission` VALUES (1299298086225002498, 1299285486804254722, 1299241863660290049, 0, '2020-08-28 18:49:38', '2020-08-28 18:49:38');
INSERT INTO `role_permission` VALUES (1299298867741278210, 1299298737512333313, 1299194681653555201, 0, '2020-08-28 18:52:45', '2020-08-28 18:52:45');
INSERT INTO `role_permission` VALUES (1299581173521522690, 1299581080902901762, 1299194681653555201, 0, '2020-08-29 13:34:31', '2020-08-29 13:34:31');

-- ----------------------------
-- Table structure for stu_info
-- ----------------------------
DROP TABLE IF EXISTS `stu_info`;
CREATE TABLE `stu_info`  (
  `id` bigint(0) NOT NULL,
  `stu_id` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '学号',
  `stu_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '姓名',
  `address` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '家庭住址',
  `political` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '政治面貌',
  `in_time` date NULL DEFAULT NULL COMMENT '入学时间',
  `out_time` date NULL DEFAULT NULL COMMENT '毕业时间',
  `college` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '学院名称',
  `prof` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '专业名称',
  `stu_class` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '行政班级',
  `academic` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '当前学历',
  `degree` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '当前学位',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  `audit_status_code` bigint(0) NULL DEFAULT NULL COMMENT '档案当前状态字典码',
  `uid` bigint(0) NULL DEFAULT NULL COMMENT '对应的用户ID',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of stu_info
-- ----------------------------
INSERT INTO `stu_info` VALUES (1, '17101020', 'lh', '四川省某某市', '中共预备党员', '2020-08-26', '2020-08-26', '计算机学院', '软件工程', '3', '本科', '学士', 0, '2020-08-26 00:00:00', '2020-08-26 00:00:00', 1, NULL);
INSERT INTO `stu_info` VALUES (243243, '749489', 'fdsadf', '四川省成都市', '中共党员', '2020-08-27', '2020-08-27', '机械学院', '数字电路', '2', '本科', '学士', 0, '2020-08-27 14:09:40', '2020-08-27 14:09:42', 1, 1299297379115679745);
INSERT INTO `stu_info` VALUES (321321, '77646', 'hfghgf', '重庆市', '中共党员', '2020-08-27', '2020-08-28', '化学学院', '应用化学', '4', '本科', '学士', 0, '2020-08-27 14:11:00', '2020-08-27 14:11:02', 1, NULL);
INSERT INTO `stu_info` VALUES (56345654, '498798', 'vxgh', '四川省广元市', '中共党员', '2020-08-27', '2020-08-29', '计算学院', '物联网', '4', '本科', '学士', 0, '2020-08-27 14:12:10', '2020-08-27 14:12:13', 1, NULL);

-- ----------------------------
-- Table structure for user
-- ----------------------------
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user`  (
  `id` bigint(0) NOT NULL COMMENT '主键ID',
  `username` varchar(128) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '账号',
  `password` varchar(128) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '密码',
  `nick_name` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '昵称',
  `real_name` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '真实姓名',
  `email` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '邮箱',
  `mobile` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '手机电话',
  `gender` varchar(16) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '性别',
  `wechat` varchar(32) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '微信号',
  `birthday` datetime(0) NULL DEFAULT NULL COMMENT '出生日期',
  `avatar` varchar(256) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '头像',
  `lasted` datetime(0) NULL DEFAULT NULL COMMENT '最后的登陆时间',
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `username`(`username`) USING BTREE,
  UNIQUE INDEX `email`(`email`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of user
-- ----------------------------
INSERT INTO `user` VALUES (2, 'admin', '123', '船长', 'test', 'test@qq.com', '135581145', '男', '', NULL, '/api/upload//images/d057d027-1e1f-4461-8ed2-230c00dc3c6b.jpg', NULL, 0, '2020-08-25 00:00:00', '2020-08-25 00:00:00');
INSERT INTO `user` VALUES (1299297379115679745, 'student', '123', NULL, '学生', NULL, '123456', '1', NULL, NULL, NULL, NULL, 0, '2020-08-28 18:46:50', '2020-08-28 18:46:50');
INSERT INTO `user` VALUES (1299297527900225538, 'medical', '123', NULL, '医护人员', NULL, '123456', '0', NULL, NULL, NULL, NULL, 0, '2020-08-28 00:00:00', '2020-08-28 00:00:00');

-- ----------------------------
-- Table structure for user_role
-- ----------------------------
DROP TABLE IF EXISTS `user_role`;
CREATE TABLE `user_role`  (
  `id` bigint(0) NOT NULL,
  `role_id` bigint(0) NULL DEFAULT NULL,
  `user_id` bigint(0) NULL DEFAULT NULL,
  `deleted` tinyint(0) NULL DEFAULT 0 COMMENT '0没有被删除，1已经被删除',
  `created` datetime(0) NULL DEFAULT NULL,
  `modified` datetime(0) NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of user_role
-- ----------------------------
INSERT INTO `user_role` VALUES (1, 1299194681653555201, 2, 0, '2020-08-22 14:25:12', '2020-08-22 14:25:14');
INSERT INTO `user_role` VALUES (1299214039373144066, 1299194681653555201, 1298295340520792065, 0, '2020-08-28 13:15:40', '2020-08-28 13:15:40');
INSERT INTO `user_role` VALUES (1299214056620122113, 1299213524224532481, 1298295618145968130, 0, '2020-08-28 13:15:44', '2020-08-28 13:15:44');
INSERT INTO `user_role` VALUES (1299297407884410881, 1299293407076278273, 1299297379115679745, 0, '2020-08-28 18:46:56', '2020-08-28 18:46:56');
INSERT INTO `user_role` VALUES (1299297528554536961, 1299241863660290049, 1299297527900225538, 0, '2020-08-28 18:47:25', '2020-08-28 18:47:25');

SET FOREIGN_KEY_CHECKS = 1;
