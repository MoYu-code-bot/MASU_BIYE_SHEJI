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
 * @since 2020-08-24
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class Activity extends BaseEntity {

    private static final long serialVersionUID=1L;

    /**
     * 档案类型
     */
    private Integer archivesTypeCode;

    /**
     * 档案当前状态字典码
     */
    private Integer auditStatusCode;

    /**
     * 学号
     */
    private String stuId;

    /**
     * 活动/竞赛名称
     */
    private String activityName;

    /**
     * 活动/竞赛等级
     */
    private String activityLevel;

    /**
     * 活动/竞赛类型
     */
    private String activityType;

    /**
     * 参与/获得时间
     */
    private Date activityTime;

    /**
     * 参与/获得证明
     */
    private String img;


}
