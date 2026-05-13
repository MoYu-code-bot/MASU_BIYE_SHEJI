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
 * @since 2020-08-22
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class Dictionary extends BaseEntity {

    private static final long serialVersionUID=1L;

    /**
     * 键
     */
    private String dictKey;

    /**
     * 字典码
     */
    private Long dictCode;

    /**
     * 值
     */
    private String dictValue;

    /**
     * 描述
     */
    private String dictDesc;


}
