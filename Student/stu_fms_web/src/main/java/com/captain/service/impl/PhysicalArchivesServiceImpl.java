package com.captain.service.impl;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.po.PhysicalArchives;
import com.captain.mapper.PhysicalArchivesMapper;
import com.captain.service.PhysicalArchivesService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author lxb
 * @since 2020-08-28
 */
@Service
public class PhysicalArchivesServiceImpl extends ServiceImpl<PhysicalArchivesMapper, PhysicalArchives> implements PhysicalArchivesService {

    @Autowired
    private PhysicalArchivesMapper physicalArchivesMapper;

    @Override
    public boolean outStock(Long id) {
        return physicalArchivesMapper.outStock(id);
    }

    @Override
    public IPage<PhysicalArchives> findListByUsernameID(Page<PhysicalArchives> page) {
        return physicalArchivesMapper.findListByUsernameID(page);
    }
}
