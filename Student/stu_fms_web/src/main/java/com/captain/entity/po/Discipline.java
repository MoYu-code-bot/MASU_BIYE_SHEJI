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
public class Discipline extends BaseEntity {

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
    private Long stuId;

    /**
     * 违纪名称
     */
    private String disName;

    /**
     * 违纪时间
     */
    private Date disTime;

    /**
     * 违纪描述
     */
    private String disDesc;

    /**
     * 处理措施
     */
    private String measure;


}
