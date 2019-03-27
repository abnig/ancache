import com.nigam.ancache.core.ANCacheManager;
import com.nigam.ancache.model.CacheElement;

public class ANCacheTest {

	public ANCacheTest() {
		
	}
	
	public static void main(String ...strings) {

		ANCacheManager<Integer, String> cache = new ANCacheManager<>(30, 5);
		
		CacheElement<K, V> i = cache.add(1, "Abhinav");
		System.out.println(i.getK());
		System.out.println(i.getV());

	}

}
