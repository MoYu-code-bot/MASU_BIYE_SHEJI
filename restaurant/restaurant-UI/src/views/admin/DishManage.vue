<template>
  <div class="dish-manage">
    <div class="page-header">
      <h2>菜品管理</h2>
      <el-button type="primary" @click="handleAdd">
        <el-icon><Plus /></el-icon>
        新增菜品
      </el-button>
    </div>

    <!-- 筛选 -->
    <div class="filter-bar">
      <el-select v-model="filterCategoryId" placeholder="按分类筛选" clearable @change="loadData" style="width: 200px;">
        <el-option label="全部分类" :value="null" />
        <el-option v-for="cat in categoryList" :key="cat.id" :label="cat.name" :value="cat.id" />
      </el-select>
    </div>

    <el-table :data="tableData" border stripe v-loading="loading" style="width: 100%">
      <el-table-column prop="id" label="ID" width="70" />
      <el-table-column prop="name" label="名称" min-width="120" />
      <el-table-column prop="categoryName" label="分类" width="100">
        <template #default="{ row }">
          {{ row.categoryName || getCategoryName(row.categoryId) }}
        </template>
      </el-table-column>
      <el-table-column prop="price" label="价格" width="90">
        <template #default="{ row }">
          ¥{{ row.price }}
        </template>
      </el-table-column>
      <el-table-column prop="spicyLevel" label="辣度" width="80">
        <template #default="{ row }">
          <span v-if="row.spicyLevel > 0">
            <span v-for="i in row.spicyLevel" :key="i">🌶️</span>
          </span>
          <span v-else>不辣</span>
        </template>
      </el-table-column>
      <el-table-column prop="isRecommended" label="推荐" width="80">
        <template #default="{ row }">
          <el-tag :type="row.isRecommended ? 'success' : 'info'" size="small">
            {{ row.isRecommended ? '是' : '否' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="isOnSale" label="上架" width="80">
        <template #default="{ row }">
          <el-tag :type="row.isOnSale ? 'success' : 'danger'" size="small">
            {{ row.isOnSale ? '上架' : '下架' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="150" fixed="right">
        <template #default="{ row }">
          <el-button size="small" type="primary" @click="handleEdit(row)">编辑</el-button>
          <el-button size="small" type="danger" @click="handleDelete(row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-pagination
      v-model:current-page="currentPage"
      v-model:page-size="pageSize"
      :total="total"
      :page-sizes="[10, 20, 50]"
      layout="total, sizes, prev, pager, next"
      @size-change="loadData"
      @current-change="loadData"
      style="margin-top: 16px; justify-content: flex-end;"
    />

    <!-- 新增/编辑对话框 -->
    <el-dialog v-model="dialogVisible" :title="isEdit ? '编辑菜品' : '新增菜品'" width="550px">
      <el-form :model="form" :rules="rules" ref="formRef" label-width="100px">
        <el-form-item label="菜品名称" prop="name">
          <el-input v-model="form.name" placeholder="请输入菜品名称" />
        </el-form-item>
        <el-form-item label="分类" prop="categoryId">
          <el-select v-model="form.categoryId" placeholder="请选择分类" style="width: 100%;">
            <el-option v-for="cat in categoryList" :key="cat.id" :label="cat.name" :value="cat.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="价格" prop="price">
          <el-input-number v-model="form.price" :min="0" :precision="2" />
        </el-form-item>
        <el-form-item label="图片URL" prop="image">
          <el-input v-model="form.image" placeholder="请输入图片URL" />
        </el-form-item>
        <el-form-item label="描述" prop="description">
          <el-input v-model="form.description" type="textarea" :rows="3" placeholder="请输入描述" />
        </el-form-item>
        <el-form-item label="辣度" prop="spicyLevel">
          <el-radio-group v-model="form.spicyLevel">
            <el-radio :value="0">不辣</el-radio>
            <el-radio :value="1">微辣</el-radio>
            <el-radio :value="2">中辣</el-radio>
            <el-radio :value="3">特辣</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="是否推荐" prop="isRecommended">
          <el-switch v-model="form.isRecommended" :active-value="1" :inactive-value="0" />
        </el-form-item>
        <el-form-item label="是否上架" prop="isOnSale">
          <el-switch v-model="form.isOnSale" :active-value="1" :inactive-value="0" />
        </el-form-item>
        <el-form-item label="排序" prop="sortOrder">
          <el-input-number v-model="form.sortOrder" :min="0" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleSubmit" :loading="submitting">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { getDishes, createDish, updateDish, deleteDish, adminGetCategories } from '@/api'

const loading = ref(false)
const tableData = ref([])
const categoryList = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const filterCategoryId = ref(null)

const dialogVisible = ref(false)
const isEdit = ref(false)
const submitting = ref(false)
const formRef = ref(null)

const defaultForm = {
  name: '',
  categoryId: null,
  price: 0,
  image: '',
  description: '',
  spicyLevel: 0,
  isRecommended: 0,
  isOnSale: 1,
  sortOrder: 0
}

const form = ref({ ...defaultForm })

const rules = {
  name: [{ required: true, message: '请输入菜品名称', trigger: 'blur' }],
  categoryId: [{ required: true, message: '请选择分类', trigger: 'change' }],
  price: [{ required: true, message: '请输入价格', trigger: 'blur' }]
}

const getCategoryName = (categoryId) => {
  const cat = categoryList.value.find(c => c.id === categoryId)
  return cat ? cat.name : ''
}

const loadCategories = async () => {
  try {
    const res = await adminGetCategories()
    categoryList.value = Array.isArray(res.data) ? res.data : []
  } catch (e) {
    // ignore
  }
}

const loadData = async () => {
  loading.value = true
  try {
    const params = {
      pageNum: currentPage.value,
      pageSize: pageSize.value
    }
    if (filterCategoryId.value) {
      params.categoryId = filterCategoryId.value
    }
    const res = await getDishes(params)
    tableData.value = res.data?.list || []
    total.value = res.data?.total || 0
  } catch (e) {
    ElMessage.error('加载菜品列表失败')
  } finally {
    loading.value = false
  }
}

const handleAdd = () => {
  isEdit.value = false
  form.value = { ...defaultForm }
  dialogVisible.value = true
}

const handleEdit = (row) => {
  isEdit.value = true
  form.value = { ...row }
  dialogVisible.value = true
}

const handleSubmit = async () => {
  await formRef.value.validate()
  submitting.value = true
  try {
    if (isEdit.value) {
      await updateDish(form.value)
      ElMessage.success('更新成功')
    } else {
      await createDish(form.value)
      ElMessage.success('创建成功')
    }
    dialogVisible.value = false
    loadData()
  } catch (e) {
    ElMessage.error(isEdit.value ? '更新失败' : '创建失败')
  } finally {
    submitting.value = false
  }
}

const handleDelete = (row) => {
  ElMessageBox.confirm(`确定删除菜品「${row.name}」吗？`, '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(async () => {
    try {
      await deleteDish(row.id)
      ElMessage.success('删除成功')
      loadData()
    } catch (e) {
      ElMessage.error('删除失败')
    }
  }).catch(() => {})
}

onMounted(() => {
  loadCategories()
  loadData()
})
</script>

<style scoped>
.dish-manage {
  padding: 10px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.page-header h2 {
  margin: 0;
  font-size: 20px;
  color: #333;
}

.filter-bar {
  margin-bottom: 16px;
}
</style>
