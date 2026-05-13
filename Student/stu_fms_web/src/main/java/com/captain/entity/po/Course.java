package com.captain.entity.po;

import com.captain.entity.po.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 *
 * </p>
 *
 * @author lianhong
 * @since 2020-08-24
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class Course extends BaseEntity {

    private static final long serialVersionUID=1L;

    /**
     * 档案类型
     */
    private Long archivesTypeCode;

    /**
     * 档案当前状态字典码
     */
    private Long auditStatusCode;

    /**
     * 学号
     */
    private String stu_id;

    /**
     * 课程名称
     */
    private String courseName;

    /**
     * 课程成绩
     */
    private Float score;

    /**
     * 课程成绩等级，优良中
     */
    private String courseLevel;

    /**
     * 排名
     */
    private Integer courseRank;

    /**
     * 任课老师
     */
    private String teacher;

    /**
     * 平时成绩
     */
    private Float usualScore;

    /**
     * 平时成绩占比
     */
    private Float usualRatio;


}
