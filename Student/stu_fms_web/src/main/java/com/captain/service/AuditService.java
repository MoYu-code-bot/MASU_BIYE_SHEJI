package com.captain.service;

import com.captain.common.Result;
import com.captain.entity.po.Audit;
import com.baomidou.mybatisplus.extension.service.IService;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author lxb
 * @since 2020-08-28
 */
public interface AuditService extends IService<Audit> {

    /**
     * 让一条待审核的记录通过审核
     * @param id
     * @return
     */
    Result passAudit(Long id);

}
