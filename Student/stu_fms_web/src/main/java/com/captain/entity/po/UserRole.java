package com.captain.entity.po;

import com.captain.entity.po.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 *
 * </p>
 *
 * @author lll
 * @since 2020-08-28
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class UserRole extends BaseEntity {

    private static final long serialVersionUID=1L;

    private Long roleId;

    private Long userId;


}
