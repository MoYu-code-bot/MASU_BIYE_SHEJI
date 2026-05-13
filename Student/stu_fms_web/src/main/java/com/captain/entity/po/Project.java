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
public class Project extends BaseEntity {

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
    private String stuId ;

    /**
     * 项目名称
     */
    private String proName;

    /**
     * 项目级别
     */
    private String proLevel;

    /**
     * 项目经费
     */
    private Float funds;

    /**
     * 项目状态
     */
    private String status;

    /**
     * 项目简介
     */
    private String descr;


}
