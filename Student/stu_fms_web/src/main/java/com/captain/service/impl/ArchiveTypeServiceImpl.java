package com.captain.service.impl;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.vo.AuditVo;
import com.captain.mapper.ArchiveTypeMapper;
import com.captain.service.ArchiveTypeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;

@Service
public class ArchiveTypeServiceImpl implements ArchiveTypeService {

    @Autowired
    private ArchiveTypeMapper archiveTypeMapper;


    @Override
    public IPage<AuditVo> list(Page page) {
        return archiveTypeMapper.list(page);
    }

    @Override
    public boolean audit(Long id) {
        return archiveTypeMapper.audit(id);
    }

    @Override
    public boolean delete(Integer id) {
        int ok = archiveTypeMapper.deleteById(id);
        if(ok <= 0){
            return false;
        }
        return true;
    }
    @Override
    public boolean myDeleteBatch(String ids) {
        return archiveTypeMapper.myDeleteBatch(ids);
    }
}
