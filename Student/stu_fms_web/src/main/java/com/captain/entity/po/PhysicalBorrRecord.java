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
 * @since 2020-08-29
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class PhysicalBorrRecord extends BaseEntity {

    private static final long serialVersionUID=1L;

    /**
     * 档案id
     */
    private String archiveId;

    /**
     * 档案名称
     */
    private String archiveName;

    /**
     * 被借阅学生id
     */
    private String stuId;

    /**
     * 被借阅学生名
     */
    private String stuName;

    /**
     * 借阅者id
     */
    private Long usernameId;

    /**
     * 借阅者
     */
    private String username;

    /**
     * 借阅时间
     */
    private Date borrTime;

    /**
     * 借阅状态，1借阅中，0已归还，-1超时
     */
    private Integer borrStatus;


}
