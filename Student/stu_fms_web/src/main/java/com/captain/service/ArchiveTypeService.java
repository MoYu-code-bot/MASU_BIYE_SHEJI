package com.captain.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.captain.entity.vo.AuditVo;

import java.util.List;

public interface ArchiveTypeService {

    IPage<AuditVo> list(Page page);

    boolean delete(Integer id);

    boolean audit(Long id);

    boolean myDeleteBatch(String ids);
}
