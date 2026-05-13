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
 * @author lxb
 * @since 2020-08-25
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class Archive extends BaseEntity {

    private static final long serialVersionUID=1L;

    private String username;

    private Long userId;

    private Integer borrCount;

    private Date cutOffTime;


}
