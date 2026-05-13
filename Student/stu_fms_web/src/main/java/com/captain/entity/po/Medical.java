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
public class Medical extends BaseEntity {

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
    private String stuId;

    /**
     * 体检结果说明
     */
    private String medicalDesc;

    /**
     * 体检结果图片(多张用分号分割)
     */
    private String img;

    /**
     * 体检时间
     */
    private Date medicalTime;

    /**
     * 体检单位
     */
    private String unit;


}
