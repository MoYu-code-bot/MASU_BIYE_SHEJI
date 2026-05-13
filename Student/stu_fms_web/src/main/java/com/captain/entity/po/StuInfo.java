package com.captain.entity.po;

import java.util.Date;
import com.captain.entity.po.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 *
 * </p>
 *
 * @author lianhong
 * @since 2020-08-26
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class StuInfo extends BaseEntity {

    private static final long serialVersionUID=1L;

    /**
     * 用户ID
     */
    private Long uid;

    /**
     * 学号
     */
    private String stuId;

    /**
     * 档案当前状态字典码
     */
    private Integer auditStatusCode;

    /**
     * 姓名
     */
    private String stuName;

    /**
     * 家庭住址
     */
    private String address;

    /**
     * 政治面貌
     */
    private String political;

    /**
     * 入学时间
     */
    private Date inTime;

    /**
     * 毕业时间
     */
    private Date outTime;

    /**
     * 学院名称
     */
    private String college;

    /**
     * 专业名称
     */
    private String prof;

    /**
     * 行政班级
     */
    private String stuClass;

    /**
     * 当前学历
     */
    private String academic;

    /**
     * 当前学位
     */
    private String degree;


}
