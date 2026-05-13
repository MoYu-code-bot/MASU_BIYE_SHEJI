package com.captain.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.po.PhysicalArchives;
import com.baomidou.mybatisplus.extension.service.IService;
import org.apache.ibatis.annotations.Param;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author lxb
 * @since 2020-08-28
 */
public interface PhysicalArchivesService extends IService<PhysicalArchives> {

    boolean outStock(Long id);

    IPage<PhysicalArchives> findListByUsernameID(Page<PhysicalArchives> page);
}
