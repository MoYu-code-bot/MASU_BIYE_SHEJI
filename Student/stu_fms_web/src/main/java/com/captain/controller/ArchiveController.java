package com.captain.controller;

        import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
        import com.captain.common.BaseRestController;
        import com.captain.common.Result;
        import com.captain.entity.po.Archive;
        import com.captain.service.ArchiveService;
        import org.springframework.web.bind.annotation.RequestMapping;
        import org.springframework.web.bind.annotation.RestController;

/**
 * 档案信息 档案审核
 */
@RestController
@RequestMapping("/archive")
public class ArchiveController extends BaseRestController<ArchiveService, Archive> {

    @Override
    protected QueryWrapper<Archive> getSearchQueryWrapper(String searchValue, Archive archive) {

        QueryWrapper<Archive> archiveQueryWrapper = new QueryWrapper<>();
        archiveQueryWrapper.like("username",searchValue);
        return archiveQueryWrapper;
    }
}
