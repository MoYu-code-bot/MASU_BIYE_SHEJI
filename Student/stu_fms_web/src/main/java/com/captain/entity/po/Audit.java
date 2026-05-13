package com.captain.entity.po;

import com.captain.entity.po.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 * 
 * </p>
 *
 * @author lxb
 * @since 2020-08-28
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class Audit extends BaseEntity {

    private static final long serialVersionUID=1L;

    /**
     * 对应的档案是哪一个类型
     */
    private Long archiveTypeCode;

    /**
     * 档案对应的ID
     */
    private Long archiveId;

    /**
     * 学号
     */
    private String stuId;

    /**
     * 是否审核,0未审核，1通过审核
     */
    private Integer isAudited;


}
