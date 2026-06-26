<template>
  <div class="store-detail" v-loading="loading">
    <!-- 门店信息头部 -->
    <div class="store-header" v-if="store">
      <div class="store-header-content">
        <h1 class="store-name">{{ store.name }}</h1>
        <div class="store-meta">
          <span class="meta-item">
            <el-icon><Location /></el-icon>
            {{ store.address }}
          </span>
          <span class="meta-item">
            <el-icon><Clock /></el-icon>
            {{ store.businessHours }}
          </span>
          <span class="meta-item">
            <el-icon><Phone /></el-icon>
            {{ store.phone }}
          </span>
        </div>
      </div>
    </div>

    <!-- 菜品分类标签 -->
    <el-tabs v-model="activeCategory" class="menu-tabs" v-if="categories.length > 0">
      <el-tab-pane
        v-for="cat in categories"
        :key="cat.id"
        :label="cat.name"
        :name="String(cat.id)"
      >
        <div class="dish-grid">
          <el-card
            v-for="dish in getDishesByCategory(cat.id)"
            :key="dish.id"
            shadow="hover"
            :class="['dish-card', isReservableCategory(cat) ? 'dish-card-clickable' : 'dish-card-preview', selectedDishId === dish.id ? 'dish-card-selected' : '']"
            @click="isReservableCategory(cat) ? selectDish(dish) : null"
          >
            <div class="dish-image-wrap">
              <img :src="dish.image || defaultDishImage" :alt="dish.name" class="dish-image" />
              <el-tag v-if="dish.isRecommended" type="danger" size="small" class="recommend-badge">推荐</el-tag>
              <div v-if="isReservableCategory(cat)" class="dish-select-hint" :class="{ 'dish-select-hint--active': selectedDishId === dish.id }">
                {{ selectedDishId === dish.id ? '✓ 已选中' : '点击预订' }}
              </div>
              <div v-else class="dish-select-hint dish-select-hint--preview">仅供预览</div>
            </div>
            <div class="dish-info">
              <h4 class="dish-name">
                {{ dish.name }}
                <span v-if="dish.spicyLevel > 0" class="spicy-level">
                  <span v-for="i in dish.spicyLevel" :key="i">🌶️</span>
                </span>
              </h4>
              <p class="dish-desc">{{ dish.description || '暂无描述' }}</p>
              <p class="dish-price">¥{{ dish.price }}</p>
            </div>
          </el-card>
        </div>
        <el-empty v-if="getDishesByCategory(cat.id).length === 0" description="该分类暂无菜品" />
      </el-tab-pane>
    </el-tabs>

    <!-- 评价区域 -->
    <div class="reviews-section" v-if="store">
      <h2 class="section-title">顾客评价</h2>
      <div class="review-list">
        <div v-for="review in reviews" :key="review.id" class="review-item">
          <div class="review-header">
            <span class="review-user">{{ review.customerName || '匿名用户' }}</span>
            <span class="review-stars">
              <span v-for="i in 5" :key="i" class="star" :class="{ active: i <= review.rating }">★</span>
            </span>
          </div>
          <p class="review-content">{{ review.content }}</p>
          <span class="review-time">{{ review.createTime }}</span>
        </div>
        <el-empty v-if="reviews.length === 0" description="暂无评价" />
      </div>
    </div>

    <!-- 预订浮动按钮 -->
    <div class="reserve-float" v-if="store">
      <el-button type="primary" size="large" round @click="goReserve">
        立即预订
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Location, Clock, Phone } from '@element-plus/icons-vue'
import { getStoreDetail, getStoreDishes, getStoreReviews, getCategories } from '@/api'

const route = useRoute()
const router = useRouter()
const loading = ref(false)

const store = ref(null)
const categories = ref([])
const dishes = ref([])
const reviews = ref([])
const activeCategory = ref('')
const selectedDish = ref(null)
const selectedDishId = ref(null)

const defaultDishImage = 'https://via.placeholder.com/200x150?text=菜品'

const getDishesByCategory = (categoryId) => {
  return dishes.value.filter(d => d.categoryId === categoryId && d.isOnSale)
}

const RESERVABLE_CATEGORY_NAME = '锅底'

const isReservableCategory = (cat) => {
  return cat.name === RESERVABLE_CATEGORY_NAME
}

const loadStoreDetail = async () => {
  loading.value = true
  const storeId = route.params.storeId
  try {
    const res = await getStoreDetail(storeId)
    store.value = res.data || res

    // 加载分类
    const catRes = await getCategories(storeId)
    categories.value = catRes.data || []
    if (categories.value.length > 0) {
      activeCategory.value = String(categories.value[0].id)
    }

    // 加载菜品
    const dishRes = await getStoreDishes(storeId)
    dishes.value = (dishRes.data || [])

    // 加载评价
    try {
      const reviewRes = await getStoreReviews(storeId, { pageNum: 1, pageSize: 20 })
      reviews.value = (reviewRes.data || []).filter(r => r.isVisible)
    } catch (e) {
      // ignore
    }
  } catch (e) {
    ElMessage.error('加载门店详情失败')
  } finally {
    loading.value = false
  }
}

const selectDish = (dish) => {
  if (selectedDishId.value === dish.id) {
    // 再次点击取消选中
    selectedDish.value = null
    selectedDishId.value = null
  } else {
    selectedDish.value = dish
    selectedDishId.value = dish.id
  }
}

const goReserve = () => {
  if (!selectedDish.value) {
    ElMessage.warning('请先选择需要预订的锅底套餐')
    return
  }
  const storeId = route.params.storeId
  router.push({
    path: `/c/reserve/${storeId}`,
    query: {
      dishId: selectedDish.value.id,
      dishName: selectedDish.value.name,
      dishPrice: selectedDish.value.price
    }
  })
}

onMounted(() => {
  loadStoreDetail()
})
</script>

<style scoped>
.store-detail {
  padding-bottom: 80px;
}

.store-header {
  background: linear-gradient(135deg, #ff6b35, #f7931e);
  color: #fff;
  padding: 30px 20px;
  border-radius: 12px;
  margin-bottom: 20px;
}

.store-name {
  font-size: 28px;
  margin: 0 0 12px 0;
}

.store-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 14px;
  opacity: 0.9;
}

.menu-tabs {
  margin-bottom: 20px;
}

.dish-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 16px;
  padding: 10px 0;
}

.dish-card {
  cursor: default;
}

.dish-card-clickable {
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.dish-card-clickable:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 20px rgba(255, 107, 53, 0.25);
}

.dish-card-preview {
  cursor: default;
  opacity: 0.85;
}

.dish-card-selected {
  border: 2px solid #ff6b35;
  box-shadow: 0 0 12px rgba(255, 107, 53, 0.35);
}

.dish-select-hint--active {
  background: rgba(255, 107, 53, 1);
  opacity: 1 !important;
}

.dish-select-hint {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(255, 107, 53, 0.85);
  color: #fff;
  text-align: center;
  font-size: 13px;
  padding: 6px 0;
  opacity: 0;
  transition: opacity 0.2s;
}

.dish-card-clickable:hover .dish-select-hint {
  opacity: 1;
}

.dish-card-preview:hover .dish-select-hint--preview {
  opacity: 1;
}

.dish-select-hint--preview {
  background: rgba(150, 150, 150, 0.85);
}

.dish-card :deep(.el-card__body) {
  padding: 0;
}

.dish-image-wrap {
  position: relative;
  width: 100%;
  height: 150px;
  overflow: hidden;
}

.dish-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.recommend-badge {
  position: absolute;
  top: 8px;
  right: 8px;
}

.dish-info {
  padding: 12px;
}

.dish-name {
  font-size: 15px;
  margin: 0 0 6px 0;
  color: #333;
  display: flex;
  align-items: center;
  gap: 4px;
}

.spicy-level {
  font-size: 12px;
}

.dish-desc {
  font-size: 12px;
  color: #999;
  margin: 0 0 8px 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.dish-price {
  font-size: 18px;
  color: #ff6b35;
  font-weight: bold;
  margin: 0;
}

.reviews-section {
  margin-top: 30px;
  padding: 20px;
  background: #fff;
  border-radius: 12px;
}

.section-title {
  font-size: 20px;
  margin: 0 0 16px 0;
  color: #333;
}

.review-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.review-item {
  padding: 16px;
  background: #f9f9f9;
  border-radius: 8px;
}

.review-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.review-user {
  font-weight: bold;
  color: #333;
}

.review-stars .star {
  color: #ddd;
  font-size: 14px;
}

.review-stars .star.active {
  color: #f7ba2a;
}

.review-content {
  font-size: 14px;
  color: #666;
  line-height: 1.6;
  margin: 0 0 8px 0;
}

.review-time {
  font-size: 12px;
  color: #bbb;
}

.reserve-float {
  position: fixed;
  bottom: 30px;
  right: 30px;
  z-index: 100;
}

.reserve-float .el-button {
  font-size: 16px;
  padding: 12px 32px;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.4);
}
</style>
